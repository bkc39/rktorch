#lang racket/base

;; The allocators Q5 and Q7 compare: today's library path and the
;; probe-local prototypes, each with the three ops the workloads use.

(require (only-in ffi/vector s64vector)
         (only-in torch/foreign/raw/collector finalizer-runs)
         (only-in torch/foreign/raw/elementwise tr-add/raw)
         (only-in torch/foreign/raw/linalg tr-matmul/raw)
         (only-in torch/foreign/raw/memory native-memory-use tr-tensor-free/checked)
         (only-in torch/foreign/raw/random tr-randn/raw)
         "prototype-allocator.rkt"
         "shim.rkt")

(provide (struct-out ops)
         workload-variants)

(define (today)
  (prototype "today: ffi/unsafe/alloc + ledger (library raw bindings)"
             (lambda (op) op)
             values
             tr-tensor-free/checked
             (lambda () (for/sum ([d (in-list (native-memory-use))]) (cdr d)))
             finalizer-runs))

;; randn, add and matmul as each variant makes them.
(struct ops (randn add matmul))

(define (today-ops)
  (ops (lambda (n) (tr-randn/raw (s64vector n n) 2)) tr-add/raw tr-matmul/raw))

(define (prototype-ops p)
  (define wrap (prototype-wrap p))
  (ops (wrap (lambda (n) (shim-randn n n))) (wrap shim-add) (wrap shim-matmul)))

(define (workload-variants #:bare? [bare? #f])
  (define makers
    (append (list stage-1 stage-2 alloc/uninterruptible)
            (if bare? (list bare) '())))
  (cons (cons (today) (today-ops))
        (for/list ([make (in-list makers)])
          (define p (make))
          (cons p (prototype-ops p)))))
