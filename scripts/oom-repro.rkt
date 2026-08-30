#lang racket/base

;; VRAM-squeeze repro for #38: balloon the card, then train so a CUDA OOM
;; strikes mid-step with a live autograd graph.
;; Run:  BALLOON=22 nix develop .#cuda --command racket scripts/oom-repro.rkt
;; (default 19 survives an idle 24 GiB card; 22-23 forces the OOM)
;; Pass: ONE clean typed exception, prompt exit. Regression signature: an
;; endless "invalid memory reference" cascade pegging a core and growing
;; host memory — the finalizer guard in raw/memory.rkt broke.

(require torch
         torch/nn
         (only-in torch/data/text
                  contiguous-blocks encode load-heart-of-darkness text->vocab)
         (only-in "../examples/racket/06-gpt.rkt" gpt))

(with-default-device 'cuda
  (manual-seed! 0)
  (display "ballooning VRAM...\n")
  (flush-output)
  ;; 1 GiB each; referenced at the end so GC can't release it mid-run
  (define balloon
    (for/list ([_ (in-range (string->number (or (getenv "BALLOON") "19")))])
      (randn 16384 16384)))
  (printf "balloon: ~a GiB held; loading corpus + training\n" (length balloon))
  (flush-output)
  (define text (load-heart-of-darkness))
  (define vocab (text->vocab text))
  (define v-size (vector-length vocab))
  (define-values (xs ys) (contiguous-blocks (encode vocab text) 64))
  (define n (car (tensor-shape xs)))
  (define net (gpt v-size 64 #:n-embd 128 #:n-head 4 #:n-layer 4))
  (define opt (adam (parameters net) #:lr 0.0003))
  (for ([start (in-range 0 (add1 (- n 64)) 64)])
    (printf "step ~a\n" (quotient start 64))
    (flush-output)
    (zero-grads! opt)
    (define loss
      (cross-entropy (reshape (net (narrow xs 0 start 64)) -1 v-size)
                     (reshape (narrow ys 0 start 64) -1)))
    (backward! loss)
    (step! opt))
  (printf "completed epoch without OOM (balloon ~a GiB — raise BALLOON to force it)\n"
          (length balloon)))
