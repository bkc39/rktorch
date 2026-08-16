#lang racket/base

;; Reproduce issue #38: hold a ~19 GiB balloon (standing in for the user's
;; concurrent V-JEPA jobs), then run the train-novel-scale loop so CUDA OOM
;; strikes mid-step with a live autograd graph. Expected today: the
;; "invalid memory reference" handler cascade instead of one clean exn.
;; Run under `timeout` — the cascade loops forever.

(require torch
         torch/nn
         (only-in torch/data/text
                  contiguous-blocks encode load-heart-of-darkness text->vocab)
         (only-in (file "/home/bkc/dev/rkt/rktorch/.claude/worktrees/gpu-memory-management/examples/racket/06-gpt.rkt")
                  gpt))

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
  (for ([start (in-range 0 (- n 64) 64)])
    (printf "step ~a\n" (quotient start 64))
    (flush-output)
    (zero-grads! opt)
    (define loss
      (cross-entropy (reshape (net (narrow xs 0 start 64)) -1 v-size)
                     (reshape (narrow ys 0 start 64) -1)))
    (backward! loss)
    (step! opt))
  (printf "completed epoch without OOM?! balloon still ~a\n" (length balloon)))
