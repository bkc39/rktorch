#lang racket/base

;; VRAM-squeeze repro for issue #38: hold a BALLOON-GiB allocation
;; (standing in for other tenants on the card), then run a
;; train-novel-scale loop so a CUDA OOM strikes mid-step with a live
;; autograd graph.
;;
;; Run:  BALLOON=22 nix develop .#cuda --command racket scripts/oom-repro.rkt
;;
;; Expected behavior WITH the #38 hardening (this tree): one clean
;; exception naming the failed op and the CUDA memory state, then a prompt
;; exit — e.g. "tr_add: CUDA out of memory. Tried to allocate ...". The
;; pre-fix failure mode this exists to detect was an infinite
;; "invalid memory reference" handler cascade that pegged a core and grew
;; host memory until the OOM killer fired; if you ever see that again, the
;; finalizer guard in torch/foreign/raw/syntax.rkt has regressed.
;; BALLOON defaults to 19 (survivable on an otherwise-idle 24 GiB card);
;; 22-23 forces the OOM.

(require torch
         torch/nn
         (only-in torch/data/text
                  contiguous-blocks encode load-heart-of-darkness text->vocab)
         (only-in "../examples/racket/06-gpt.rkt" gpt))

(with-default-device 'cuda
  (manual-seed! 0)
  (printf "ballooning VRAM...\n")
  (flush-output)
  ;; BALLOON x 1 GiB, retained for the whole run (referenced at the end).
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
  ;; end bound inclusive of the final window (the train-gpt convention).
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
