#lang racket/base

;; Safe tensor operations built on the raw layer + wrapper struct.  These are
;; the implementation behind the contracts in ../foreign.rkt.

(require ffi/vector
         "error.rkt"
         "raw/global.rkt"
         "raw/random.rkt"
         "raw/tensor.rkt"
         "structs.rkt")

(provide torch-version
         manual-seed!
         randn
         tensor-numel
         tensor-shape
         tensor->vector
         tensor->list
         tensor->string
         tensor->repr)

(define (torch-version)
  (tr-version/raw))

(define (manual-seed! seed)
  (check-ok (tr-manual-seed/raw seed) 'manual-seed!)
  (void))

(define (randn . dims)
  (define h (tr-randn/raw (list->s64vector dims) (length dims)))
  (unless h
    (error 'randn "randn failed: ~a" (tr-last-error/raw)))
  (wrap-tensor h))

(define (tensor-numel t)
  (define-values (rc n) (tr-tensor-numel/raw t))
  (check-ok rc 'tensor-numel)
  n)

;; Cached at wrap time, so no C round-trip.
(define (tensor-shape t)
  (tensor-impl-shape t))

(define (tensor->vector t)
  (define numel (tensor-numel t))
  (define out (make-f32vector numel))
  (define-values (rc _numel) (tr-tensor-copy-data/raw t numel out))
  (check-ok rc 'tensor->vector)
  out)

(define (tensor->list t)
  (f32vector->list (tensor->vector t)))

;; ATen's C++ `operator<<` text (the libtorch-native printer).
(define (tensor->string t)
  (handle->string (tensor-handle t)))

;; The PyTorch `repr` text -- identical to what the REPL prints (see structs.rkt).
(define (tensor->repr t)
  (handle->repr (tensor-handle t) (tensor-shape t)))
