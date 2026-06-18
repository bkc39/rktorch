#lang racket/base

;; Device-placement facade tests. The CPU behaviors run everywhere; the CUDA
;; round-trip self-skips unless a real device is visible, so the same suite
;; verifies the GPU when run on a CUDA host (see the cuda-verify flake app).

(module+ test
  (require rackunit
           "../main.rkt")

  (test-case "default device is cpu"
    ;; reset defensively (mirrors the C++ DefaultsToCpu): if a later CUDA case
    ;; ever leaks the default, this case shouldn't depend on source order.
    (set-default-device! 'cpu)
    (check-equal? (default-device) 'cpu))

  (test-case "cuda queries have sane types"
    (check-true (boolean? (cuda-available?)))
    (check-pred exact-nonnegative-integer? (cuda-device-count))
    ;; the count is positive exactly when a device is available
    (check-equal? (> (cuda-device-count) 0) (cuda-available?)))

  (test-case "new tensors and to-device land on cpu"
    (define t (zeros 2 2))
    (check-equal? (tensor-device t) 'cpu)
    (define c (to-device t 'cpu))
    (check-equal? (tensor-device c) 'cpu)
    (check-equal? (tensor->list c) '(0.0 0.0 0.0 0.0)))

  (test-case "set-default-device! round-trips on cpu"
    (set-default-device! 'cpu)
    (check-equal? (default-device) 'cpu)
    (check-equal? (tensor-device (ones 3)) 'cpu))

  (test-case "requesting an unavailable cuda device errors"
    (unless (cuda-available?)
      (check-exn exn:fail? (lambda () (set-default-device! 'cuda)))
      ;; the rejected set leaves the default untouched
      (check-equal? (default-device) 'cpu)))

  ;; The CUDA cases are always registered (so the test count is hardware-stable
  ;; and the cases are visible in the run output); their bodies are guarded by
  ;; `when` and only execute on a real CUDA host, exactly like the CPU-gated
  ;; "requesting an unavailable cuda device errors" case above.
  (test-case "out-of-range cuda ordinal errors"
    ;; the C++ set_default_device validates index < device_count(); this is the
    ;; only place that rejection path has coverage on a real GPU host.
    (when (cuda-available?)
      (check-exn exn:fail?
                 (lambda () (set-default-device! (list 'cuda 9999))))
      (check-equal? (default-device) 'cpu)))

  (test-case "cuda round-trip"
    ;; with-default-device restores the prior default even if a GPU op raises
    ;; (matmul/tensor->list can throw), so CUDA can't leak onto later tests.
    (when (cuda-available?)
      (check-true (> (cuda-device-count) 0))
      (set-default-device! 'cpu)
      (with-default-device 'cuda
        (check-equal? (default-device) '(cuda 0))
        (define g (zeros 2 2))
        (check-equal? (tensor-device g) '(cuda 0))
        (define back (to-device g 'cpu))
        (check-equal? (tensor-device back) 'cpu)
        (check-equal? (tensor->list back) '(0.0 0.0 0.0 0.0))
        ;; a GPU matmul should match the CPU result
        (define a (to-device (tensor '((1 2) (3 4))) 'cuda))
        (define b (to-device (tensor '((5 6) (7 8))) 'cuda))
        (check-equal? (tensor->list (to-device (matmul a b) 'cpu))
                      '(19.0 22.0 43.0 50.0)))
      (check-equal? (default-device) 'cpu)))

  (test-case "with-default-device restores the prior default (return + raise)"
    (set-default-device! 'cpu)
    ;; normal return restores
    (with-default-device 'cpu (check-equal? (default-device) 'cpu))
    (check-equal? (default-device) 'cpu)
    ;; restored even when the body raises — the dynamic-wind guarantee a
    ;; hand-rolled set/reset would drop
    (check-exn exn:fail? (lambda () (with-default-device 'cpu (error "boom"))))
    (check-equal? (default-device) 'cpu)
    ;; nesting restores to the *enclosing* device, not the pre-outer default
    (when (cuda-available?)
      (with-default-device 'cuda
        (check-equal? (default-device) '(cuda 0))
        (with-default-device 'cpu (check-equal? (default-device) 'cpu))
        (check-equal? (default-device) '(cuda 0)))
      (check-equal? (default-device) 'cpu)
      (check-exn exn:fail?
                 (lambda () (with-default-device 'cuda (error "boom"))))
      (check-equal? (default-device) 'cpu))))
