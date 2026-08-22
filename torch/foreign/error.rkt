#lang racket/base

;; Shared error helpers for the safe layer.  torchrkt C functions either
;; return an integer status (0 ok) with `tr_last_error` holding the message,
;; or a handle that is NULL on failure. Failures the C side classified as
;; allocation exhaustion raise the typed exn:fail:rktorch:oom, so callers
;; catch OOM by TYPE — never by regexing messages.

(require (only-in ffi/unsafe/atomic call-as-atomic)
         (only-in "raw/global.rkt" tr-last-error-kind/raw tr-last-error/raw))

(provide check-ok
         check-handle
         (struct-out exn:fail:rktorch:oom))

(struct exn:fail:rktorch:oom exn:fail ())

;; Atomic so another thread's torchrkt failure can't land between the two
;; FFI reads — the (message, kind) pair raised from stays consistent.
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
