#lang racket/base

;; CPU behaviors run everywhere; the CUDA/MPS cases are `when`-guarded, so
;; the suite verifies the GPU for real on a CUDA host or Apple Silicon.

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
    (check-true (device? (mps-device)))
    (check-equal? (device-type (mps-device)) 'mps)
    (check-equal? (device-index (mps-device)) 0)
    (check-exn exn:fail? (lambda () (device 'metal 0)))
    (check-exn exn:fail? (lambda () (device 'mps 1)))
    (check-exn exn:fail? (lambda () (device 'cuda -1)))
    (check-exn exn:fail? (lambda () (device 'cpu 1)))
    (check-equal? (format "~a" (cpu-device)) "#<device cpu>")
    (check-equal? (format "~a" (cuda-device 1)) "#<device cuda:1>")
    (check-equal? (format "~a" (mps-device)) "#<device mps>"))

  (test-case "device structs are field-wise equal? and hash keys"
    (check-equal? (cpu-device) (cpu-device))
    (check-equal? (cuda-device 1) (device 'cuda 1))
    (check-false (equal? (cuda-device 0) (cuda-device 1)))
    (define h (make-hash))
    (hash-update! h (cpu-device) add1 0)
    (hash-update! h (cpu-device) add1 0)
    (hash-update! h (cuda-device 3) add1 0)
    (check-equal? (hash-ref h (cpu-device)) 2)
    (check-equal? (hash-ref h (cuda-device 3)) 1)
    (check-equal? (hash-count h) 2)
    (check-equal? (mps-device) (device 'mps))
    (check-false (equal? (mps-device) (cpu-device))))

  (test-case "tensor #:device places construction; cuda-if-available picks"
    (set-default-device! 'cpu)
    (define t (tensor '(1 2 3) #:device (cpu-device)))
    (check-equal? (tensor-device t) (cpu-device))
    (check-equal? (tensor-device (tensor '(1 2) #:device 'cpu)) (cpu-device))
    (check-equal? (default-device) (cpu-device))
    (check-true (requires-grad?
                 (tensor '(1.0) #:device (cpu-device) #:requires-grad? #t)))
    (check-equal? (device-type (cuda-if-available))
                  (if (cuda-available?) 'cuda 'cpu))
    (check-equal? (device-index (cuda-if-available)) 0)
    (check-equal? (device-type (mps-if-available))
                  (if (mps-available?) 'mps 'cpu))
    (check-equal? (device-index (mps-if-available)) 0)
    (when (cuda-available?)
      (define g (tensor '(1 2 3) #:device (cuda-device)))
      (check-equal? (tensor-device g) (cuda-device 0))
      (check-equal? (tensor->list (to-device g (cpu-device))) '(1 2 3))
      (check-equal? (default-device) (cpu-device))
      ;; an explicitly-CPU tensor under a CUDA default lands on CPU
      (with-default-device (cuda-device)
        (check-equal? (tensor-device (tensor '(4 5) #:device (cpu-device)))
                      (cpu-device)))
      (check-equal? (default-device) (cpu-device))))

  (test-case "cuda allocator gauges (#51)"
    ;; the device guard rejects before any FFI call — hardware-independent
    (check-exn #rx"expected a CUDA device"
               (lambda () (cuda-memory-stats (cpu-device))))
    (cond
      [(cuda-available?)
       (define g (tensor '(1 2 3 4) #:device (cuda-device)))
       (define stats (cuda-memory-stats))
       (define (stat k) (cdr (assq k stats)))
       (check-true (> (stat 'allocated) 0))
       (check-true (>= (stat 'peak-allocated) (stat 'allocated)))
       (check-true (>= (stat 'reserved) (stat 'allocated)))
       (cuda-empty-cache!)
       (check-true (<= (cdr (assq 'reserved (cuda-memory-stats)))
                       (stat 'reserved)))
       ;; keep g live through the gauge reads
       (check-equal? (tensor-device g) (cuda-device 0))]
      [else
       (check-exn exn:fail? (lambda () (cuda-memory-stats)))
       (check-not-exn cuda-empty-cache!)]))

  (test-case "device arguments accept structs and legacy forms alike"
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
    (set-default-device! 'cpu)
    (check-equal? (default-device) (cpu-device)))

  (test-case "cuda queries have sane types"
    (check-true (boolean? (cuda-available?)))
    (check-pred exact-nonnegative-integer? (cuda-device-count))
    (check-equal? (> (cuda-device-count) 0) (cuda-available?)))

  (test-case "mps queries have sane types"
    (check-true (boolean? (mps-available?)))
    (check-not-exn mps-empty-cache!))

  (test-case "accelerator-if-available prefers cuda, then mps, then cpu"
    (define picked (accelerator-if-available))
    (check-true (device? picked))
    (check-equal? (device-type picked)
                  (cond
                    [(cuda-available?) 'cuda]
                    [(mps-available?) 'mps]
                    [else 'cpu]))
    (check-equal? (device-index picked) 0)
    (check-equal? (tensor-device (tensor '(1 2 3) #:device picked)) picked))

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
      (check-equal? (default-device) (cpu-device))))

  (test-case "requesting an unavailable mps device errors"
    (unless (mps-available?)
      (check-exn exn:fail? (lambda () (set-default-device! 'mps)))
      (check-equal? (default-device) (cpu-device))))

  (test-case "out-of-range cuda ordinal errors"
    (when (cuda-available?)
      (check-exn exn:fail?
                 (lambda () (set-default-device! (list 'cuda 9999))))
      (check-equal? (default-device) (cpu-device))))

  (test-case "cuda round-trip"
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
        ;; cuBLAS has no int64 matmul: floats work, int64 must reject
        (define a (to-device (tensor '((1.0 2.0) (3.0 4.0))) 'cuda))
        (define b (to-device (tensor '((5.0 6.0) (7.0 8.0))) 'cuda))
        (check-equal? (tensor->list (to-device (matmul a b) 'cpu))
                      '(19.0 22.0 43.0 50.0))
        (check-exn exn:fail?
                   (lambda ()
                     (matmul (to-device (tensor '((1 2) (3 4))) 'cuda)
                             (to-device (tensor '((1 2) (3 4))) 'cuda)))))
      (check-equal? (default-device) (cpu-device))))

  (test-case "tranche-3 ops run on cuda (gelu, embedding, layer-norm, mask)"
    (when (cuda-available?)
      (set-default-device! 'cpu)
      (with-default-device 'cuda
        (define g
          (to-device (gelu (to-device (tensor '(0.0 1.0 -1.0)) 'cuda)) 'cpu))
        (check-= (cadr (tensor->list g)) 0.841345 1e-5)
        (define w (to-device (reshape (arange 1 9) 4 2) 'cuda))
        (define idx (to-device (to-dtype (tensor '(2 0 2)) 'int64) 'cuda))
        (check-equal? (tensor->list (to-device (embedding idx w) 'cpu))
                      '(5.0 6.0 1.0 2.0 5.0 6.0))
        (define x (to-device (tensor '((1.0 2.0 3.0) (4.0 6.0 8.0))) 'cuda))
        (define ln (layer-norm x 3 #:weight (ones 3) #:bias (zeros 3)))
        (check-equal? (tensor-device ln) (cuda-device 0))
        (check-= (car (tensor->list (to-device ln 'cpu))) -1.2247 1e-4)
        (define mask (eq (tril (ones 2 2)) 0))
        (check-equal? (tensor-device mask) (cuda-device 0))
        (check-equal? (tensor->list
                       (to-device (masked-fill (ones 2 2) mask 0) 'cpu))
                      '(1.0 0.0 1.0 1.0)))
      (check-equal? (default-device) (cpu-device))))

  (test-case "mps round-trip"
    (when (mps-available?)
      (set-default-device! 'cpu)
      (with-default-device 'mps
        (check-equal? (default-device) (mps-device))
        (define g (zeros 2 2))
        (check-equal? (tensor-device g) (mps-device))
        (define back (to-device g 'cpu))
        (check-equal? (tensor-device back) (cpu-device))
        (check-equal? (tensor->list back) '(0.0 0.0 0.0 0.0))
        (define a (to-device (tensor '((1.0 2.0) (3.0 4.0))) 'mps))
        (define b (to-device (tensor '((5.0 6.0) (7.0 8.0))) 'mps))
        (check-equal? (tensor->list (to-device (matmul a b) 'cpu))
                      '(19.0 22.0 43.0 50.0))
        (check-equal? (tensor-device (tensor '(4 5) #:device (cpu-device)))
                      (cpu-device))
        (manual-seed! 42)
        (define r1 (tensor->list (to-device (randn 4) 'cpu)))
        (manual-seed! 42)
        (check-equal? (tensor->list (to-device (randn 4) 'cpu)) r1)
        (mps-empty-cache!))
      (check-equal? (default-device) (cpu-device))))

  (test-case "tranche-3 ops run on mps (gelu, embedding, layer-norm, mask)"
    (when (mps-available?)
      (set-default-device! 'cpu)
      (with-default-device 'mps
        (define g
          (to-device (gelu (to-device (tensor '(0.0 1.0 -1.0)) 'mps)) 'cpu))
        (check-= (cadr (tensor->list g)) 0.841345 1e-5)
        (define w (to-device (reshape (arange 1 9) 4 2) 'mps))
        (define idx (to-device (to-dtype (tensor '(2 0 2)) 'int64) 'mps))
        (check-equal? (tensor->list (to-device (embedding idx w) 'cpu))
                      '(5.0 6.0 1.0 2.0 5.0 6.0))
        (define x (to-device (tensor '((1.0 2.0 3.0) (4.0 6.0 8.0))) 'mps))
        (define ln (layer-norm x 3 #:weight (ones 3) #:bias (zeros 3)))
        (check-equal? (tensor-device ln) (mps-device))
        (check-= (car (tensor->list (to-device ln 'cpu))) -1.2247 1e-4)
        (define mask (eq (tril (ones 2 2)) 0))
        (check-equal? (tensor-device mask) (mps-device))
        (check-equal? (tensor->list
                       (to-device (masked-fill (ones 2 2) mask 0) 'cpu))
                      '(1.0 0.0 1.0 1.0)))
      (check-equal? (default-device) (cpu-device))))

  (test-case "with-default-device restores the prior default (return + raise)"
    (set-default-device! 'cpu)
    (with-default-device 'cpu (check-equal? (default-device) (cpu-device)))
    (check-equal? (default-device) (cpu-device))
    (check-exn exn:fail? (lambda () (with-default-device 'cpu (error "boom"))))
    (check-equal? (default-device) (cpu-device))
    (when (cuda-available?)
      (with-default-device 'cuda
        (check-equal? (default-device) (cuda-device 0))
        (with-default-device 'cpu (check-equal? (default-device) (cpu-device)))
        (check-equal? (default-device) (cuda-device 0)))
      (check-equal? (default-device) (cpu-device))
      (check-exn exn:fail?
                 (lambda () (with-default-device 'cuda (error "boom"))))
      (check-equal? (default-device) (cpu-device)))))
