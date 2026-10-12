#lang racket/base

;; The allocators Q5 and Q7 compare, each with the three ops the workloads use.

(require (only-in ffi/vector s64vector)
         (only-in torch/foreign/raw/collector finalizer-runs)
         (only-in torch/foreign/raw/elementwise tr-add/raw)
         (only-in torch/foreign/raw/linalg tr-matmul/raw)
         (only-in torch/foreign/raw/memory native-memory-use tr-tensor-free/checked)
         (only-in torch/foreign/raw/random tr-randn/raw)
         "prototype-allocator.rkt"
         "shim.rkt")

(provide (struct-out ops)
         (struct-out variant)
         variant-makers)

(struct ops (randn add matmul))

(struct variant (prototype ops global-ledger?))

(define (today)
  (variant
   (prototype "today: ffi/unsafe/alloc + ledger + pressure gate (library raw bindings)"
              (lambda (op) op)
              values
              tr-tensor-free/checked
              (lambda () (for/sum ([d (in-list (native-memory-use))]) (cdr d)))
              finalizer-runs)
   (ops (lambda (n) (tr-randn/raw (s64vector n n) 2)) tr-add/raw tr-matmul/raw)
   #t))

(define ((fresh make))
  (define p (make))
  (define wrap (prototype-wrap p))
  (variant p
           (ops (wrap (lambda (n) (shim-randn n n))) (wrap shim-add) (wrap shim-matmul))
           #f))

(define (variant-makers #:bare? [bare? #f])
  (cons today
        (map fresh
             (append (list stage-1 stage-1/gate stage-2 alloc/uninterruptible)
                     (if bare? (list bare) '())))))
