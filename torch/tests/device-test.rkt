#lang racket/base

;; Device-placement facade tests. The CPU behaviors run everywhere; the CUDA
;; round-trip self-skips unless a real device is visible, so the same suite
;; verifies the GPU when run on a CUDA host (see the cuda-verify flake app).

(module+ test
  (require rackunit
           "../main.rkt")

  (test-case "device structs: predicate, accessors, guard, printing"
    (check-true (device? (cpu-device)))
    (check-true (device? (cuda-device 1)))
    (check-false (device? 'cpu))
    (check-equal? (device-type (cpu-device)) 'cpu)
    (check-equal? (device-type (cuda-device)) 'cuda)
    (check-equal? (device-index (cuda-device 2)) 2)
    (check-equal? (device-index (cuda-device)) 0)
    ;; the raw constructor validates — malformed devices unrepresentable
    (check-exn exn:fail? (lambda () (device 'mps 0)))
    (check-exn exn:fail? (lambda () (device 'cuda -1)))
    ;; one CPU: nonzero index rejected here rather than inconsistently
    ;; downstream (C++ set-default rejects it; to-device drops it)
    (check-exn exn:fail? (lambda () (device 'cpu 1)))
    ;; torch.device-style printing
    (check-equal? (format "~a" (cpu-device)) "#<device cpu>")
    (check-equal? (format "~a" (cuda-device 1)) "#<device cuda:1>"))

  (test-case "device structs are field-wise equal? and hash keys"
    ;; transparent structs: distinct instances with equal fields are equal?
    ;; and collide as equal-hash keys — the property the #37 memory ledger
    ;; relies on for its per-device buckets.
    (check-equal? (cpu-device) (cpu-device))
    (check-equal? (cuda-device 1) (device 'cuda 1))
    (check-false (equal? (cuda-device 0) (cuda-device 1)))
    (define h (make-hash))
    (hash-update! h (cpu-device) add1 0)
    (hash-update! h (cpu-device) add1 0)
    (hash-update! h (cuda-device 3) add1 0)
    (check-equal? (hash-ref h (cpu-device)) 2)
    (check-equal? (hash-ref h (cuda-device 3)) 1)
    (check-equal? (hash-count h) 2))

  (test-case "tensor #:device places construction; cuda-if-available picks"
    ;; the smart constructor's keyword places via construct-then-move
    ;; (never by touching the process-global default device) — the
    ;; ambient default is untouched after.
    (set-default-device! 'cpu)
    (define t (tensor '(1 2 3) #:device (cpu-device)))
    (check-equal? (tensor-device t) (cpu-device))
    (check-equal? (tensor-device (tensor '(1 2) #:device 'cpu)) (cpu-device))
    (check-equal? (default-device) (cpu-device))
    ;; requires-grad composes with #:device
    (check-true (requires-grad?
                 (tensor '(1.0) #:device (cpu-device) #:requires-grad? #t)))
    ;; cuda-if-available: the promoted pick-device idiom. The expected
    ;; side is derived from device-TYPE symbols, not by mirroring the
    ;; implementation's own constructor expression — a swapped branch in
    ;; cuda-if-available must produce a mismatch here, not reproduce it.
    (check-equal? (device-type (cuda-if-available))
                  (if (cuda-available?) 'cuda 'cpu))
    (check-equal? (device-index (cuda-if-available)) 0)
    (when (cuda-available?)
      (define g (tensor '(1 2 3) #:device (cuda-device)))
      (check-equal? (tensor-device g) (cuda-device 0))
      ;; the payload survives the CPU->CUDA construction leg, not just the
      ;; device tag (marshalled back through an explicit move to CPU)
      (check-equal? (tensor->list (to-device g (cpu-device))) '(1.0 2.0 3.0))
      (check-equal? (default-device) (cpu-device))
      ;; placement is passed into native construction, so an explicitly-CPU
      ;; tensor under a CUDA default lands on CPU (no host->GPU->CPU
      ;; bounce). with-default-device restores on ANY exit, so a failing
      ;; check can't leak a CUDA default onto later test-cases.
      (with-default-device (cuda-device)
        (check-equal? (tensor-device (tensor '(4 5) #:device (cpu-device)))
                      (cpu-device)))
      (check-equal? (default-device) (cpu-device))))

  (test-case "cuda allocator gauges (#51)"
    (cond
      [(cuda-available?)
       ;; with a live GPU tensor the gauges are sane: allocated positive,
       ;; peak >= allocated, reserved >= allocated
       (define g (tensor '(1 2 3 4) #:device (cuda-device)))
       (define stats (cuda-memory-stats))
       (define (stat k) (cdr (assq k stats)))
       (check-true (> (stat 'allocated) 0))
       (check-true (>= (stat 'peak-allocated) (stat 'allocated)))
       (check-true (>= (stat 'reserved) (stat 'allocated)))
       ;; empty-cache succeeds and never raises reserved
       (cuda-empty-cache!)
       (check-true (<= (cdr (assq 'reserved (cuda-memory-stats)))
                       (stat 'reserved)))
       ;; keep g live through the gauge reads
       (check-equal? (tensor-device g) (cuda-device 0))]
      [else
       ;; without CUDA the stats error cleanly and empty-cache! no-ops
       (check-exn exn:fail? (lambda () (cuda-memory-stats)))
       (check-not-exn cuda-empty-cache!)]))

  (test-case "device arguments accept structs and legacy forms alike"
    ;; the accept-both contract: every device-taking entry point normalizes
    ;; struct and legacy inputs identically; queries return structs.
    (set-default-device! (cpu-device))
    (check-equal? (default-device) (cpu-device))
    (set-default-device! 'cpu)
    (check-equal? (default-device) (cpu-device))
    (define t (zeros 2))
    (check-equal? (tensor-device (to-device t (cpu-device))) (cpu-device))
    (check-equal? (tensor-device (to-device t 'cpu)) (cpu-device))
    (with-default-device (cpu-device)
      (check-equal? (default-device) (cpu-device))))

  (test-case "default device is cpu"
    ;; reset defensively (mirrors the C++ DefaultsToCpu): if a later CUDA case
    ;; ever leaks the default, this case shouldn't depend on source order.
    (set-default-device! 'cpu)
    (check-equal? (default-device) (cpu-device)))

  (test-case "cuda queries have sane types"
    (check-true (boolean? (cuda-available?)))
    (check-pred exact-nonnegative-integer? (cuda-device-count))
    ;; the count is positive exactly when a device is available
    (check-equal? (> (cuda-device-count) 0) (cuda-available?)))

  (test-case "new tensors and to-device land on cpu"
    (define t (zeros 2 2))
    (check-equal? (tensor-device t) (cpu-device))
    (define c (to-device t 'cpu))
    (check-equal? (tensor-device c) (cpu-device))
    (check-equal? (tensor->list c) '(0.0 0.0 0.0 0.0)))

  (test-case "set-default-device! round-trips on cpu"
    (set-default-device! 'cpu)
    (check-equal? (default-device) (cpu-device))
    (check-equal? (tensor-device (ones 3)) (cpu-device)))

  (test-case "requesting an unavailable cuda device errors"
    (unless (cuda-available?)
      (check-exn exn:fail? (lambda () (set-default-device! 'cuda)))
      ;; the rejected set leaves the default untouched
      (check-equal? (default-device) (cpu-device))))

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
      (check-equal? (default-device) (cpu-device))))

  (test-case "cuda round-trip"
    ;; with-default-device restores the prior default even if a GPU op raises
    ;; (matmul/tensor->list can throw), so CUDA can't leak onto later tests.
    (when (cuda-available?)
      (check-true (> (cuda-device-count) 0))
      (set-default-device! 'cpu)
      (with-default-device 'cuda
        (check-equal? (default-device) (cuda-device 0))
        (define g (zeros 2 2))
        (check-equal? (tensor-device g) (cuda-device 0))
        (define back (to-device g 'cpu))
        (check-equal? (tensor-device back) (cpu-device))
        (check-equal? (tensor->list back) '(0.0 0.0 0.0 0.0))
        ;; a GPU matmul should match the CPU result
        (define a (to-device (tensor '((1 2) (3 4))) 'cuda))
        (define b (to-device (tensor '((5 6) (7 8))) 'cuda))
        (check-equal? (tensor->list (to-device (matmul a b) 'cpu))
                      '(19.0 22.0 43.0 50.0)))
      (check-equal? (default-device) (cpu-device))))

  (test-case "tranche-3 ops run on cuda (gelu, embedding, layer-norm, mask)"
    ;; the GPU half of the #22 done-criterion until the 06-gpt capstone's
    ;; CUDA parity pass exists: every transformer primitive executes on the
    ;; device and lands the same values as CPU.
    (when (cuda-available?)
      (set-default-device! 'cpu)
      (with-default-device 'cuda
        ;; gelu (hand-written path)
        (define g (to-device (gelu (to-device (tensor '(0 1 -1)) 'cuda)) 'cpu))
        (check-= (cadr (tensor->list g)) 0.841345 1e-5)
        ;; embedding: int64 indices gathered on the device
        (define w (to-device (reshape (arange 1 9) 4 2) 'cuda))
        (define idx (to-device (to-dtype (tensor '(2 0 2)) 'int64) 'cuda))
        (check-equal? (tensor->list (to-device (embedding idx w) 'cpu))
                      '(5.0 6.0 1.0 2.0 5.0 6.0))
        ;; layer-norm with affine params on the device
        (define x (to-device (tensor '((1 2 3) (4 6 8))) 'cuda))
        (define ln (layer-norm x 3 #:weight (ones 3) #:bias (zeros 3)))
        (check-equal? (tensor-device ln) (cuda-device 0))
        (check-= (car (tensor->list (to-device ln 'cpu))) -1.2247 1e-4)
        ;; the causal-mask chain: tril -> eq -> masked-fill, all on cuda
        (define mask (eq (tril (ones 2 2)) 0))
        (check-equal? (tensor-device mask) (cuda-device 0))
        (check-equal? (tensor->list
                       (to-device (masked-fill (ones 2 2) mask 0) 'cpu))
                      '(1.0 0.0 1.0 1.0)))
      (check-equal? (default-device) (cpu-device))))

  (test-case "with-default-device restores the prior default (return + raise)"
    (set-default-device! 'cpu)
    ;; normal return restores
    (with-default-device 'cpu (check-equal? (default-device) (cpu-device)))
    (check-equal? (default-device) (cpu-device))
    ;; restored even when the body raises — the dynamic-wind guarantee a
    ;; hand-rolled set/reset would drop
    (check-exn exn:fail? (lambda () (with-default-device 'cpu (error "boom"))))
    (check-equal? (default-device) (cpu-device))
    ;; nesting restores to the *enclosing* device, not the pre-outer default
    (when (cuda-available?)
      (with-default-device 'cuda
        (check-equal? (default-device) (cuda-device 0))
        (with-default-device 'cpu (check-equal? (default-device) (cpu-device)))
        (check-equal? (default-device) (cuda-device 0)))
      (check-equal? (default-device) (cpu-device))
      (check-exn exn:fail?
                 (lambda () (with-default-device 'cuda (error "boom"))))
      (check-equal? (default-device) (cpu-device)))))
