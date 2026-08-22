#lang racket/base

(require (only-in ffi/unsafe/atomic call-as-atomic)
         (only-in "raw/global.rkt" tr-last-error-kind/raw tr-last-error/raw))

(provide check-ok
         check-handle
         (struct-out exn:fail:rktorch:oom))

(struct exn:fail:rktorch:oom exn:fail ())

;; Atomic so the (message, kind) pair comes from ONE failure, not two.
(define (last-failure)
  (call-as-atomic
   (lambda () (values (tr-last-error/raw) (tr-last-error-kind/raw)))))

(define (raise-torch-failure who describe)
  (define-values (message kind) (last-failure))
  (define full (format "~a: ~a" who (describe message)))
  (when (= 1 kind)
    (raise (exn:fail:rktorch:oom full (current-continuation-marks))))
  (raise (exn:fail full (current-continuation-marks))))

(define (check-ok rc who)
  (unless (zero? rc)
    (raise-torch-failure
     who
     (lambda (m) (format "FFI call failed (rc=~a): ~a" rc m)))))

(define (check-handle who h)
  (unless h
    (raise-torch-failure
     who
     (lambda (m) (format "torchrkt returned NULL handle: ~a" m))))
  h)
