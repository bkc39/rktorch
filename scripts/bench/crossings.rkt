#lang racket/base

(require (only-in ffi/unsafe
                  _double _fun _int _int64 _list _pointer _ptr ffi-lib get-ffi-obj)
         (only-in ffi/vector make-s64vector _s64vector)
         (only-in torch randn)
         (only-in torch/foreign/raw/syntax _Tensor)
         (only-in torch/foreign/structs tensor-handle)
         (only-in "harness.rkt" bench-case)
         (only-in "shim.rkt" shim-library))

(provide crossing-cases)

(define (shim name type)
  (get-ffi-obj name shim-library type))

(define fmax (get-ffi-obj "fmax" #f (_fun _double _double -> _double)))
(define fmax/blocking
  (get-ffi-obj "fmax" #f (_fun #:blocking? #t _double _double -> _double)))
(define last-error-kind (shim "tr_last_error_kind" (_fun -> _int)))
(define ndim/pointer (shim "tr_tensor_ndim" (_fun _pointer _s64vector -> _int)))
(define ndim/tagged (shim "tr_tensor_ndim" (_fun _Tensor _s64vector -> _int)))
(define ndim/ptr-o
  (shim "tr_tensor_ndim"
        (_fun _Tensor (out : (_ptr o _int64)) -> (rc : _int) -> (+ rc out))))
(define shape/s64vector
  (shim "tr_tensor_shape" (_fun _Tensor _int64 _s64vector _s64vector -> _int)))
(define shape/list
  (shim "tr_tensor_shape" (_fun _Tensor _int64 (_list i _int64) _s64vector -> _int)))

(define (crossing-cases #:reps [reps 200000])
  (define t (randn 8 8))
  (define h (tensor-handle t))
  (define out (make-s64vector 1))
  (define dims (make-s64vector 4))
  (define dim-list '(0 0 0 0))
  (define ((loop f) n)
    (for/fold ([acc 0]) ([_ (in-range n)])
      (+ acc (f))))
  (define (case name f)
    (bench-case name (loop f) #:reps reps))
  (list
   (case 'scalar-double (lambda () (if (> (fmax 1.5 2.5) 2.0) 1 0)))
   (case 'scalar-int (lambda () (last-error-kind)))
   (case 'pointer (lambda () (ndim/pointer h out)))
   (case 'tagged-pointer (lambda () (ndim/tagged h out)))
   (case 'tagged-struct (lambda () (ndim/tagged t out)))
   (case 'ptr-o (lambda () (ndim/ptr-o h)))
   (case 's64vector (lambda () (shape/s64vector h 4 dims out)))
   (case 'list-4 (lambda () (shape/list h 4 dim-list out)))
   (case 'blocking (lambda () (if (> (fmax/blocking 1.5 2.5) 2.0) 1 0)))))
