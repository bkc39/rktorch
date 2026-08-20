#lang racket/base

;; Shared error helpers for the safe layer.  torchrkt C functions either return
;; an integer status (0 ok) with `tr_last_error` holding the message, or a
;; handle that is NULL on failure.
;;
;; Failures the C side classified as allocation exhaustion
;; (tr_last_error_kind = 1: CUDA's c10::OutOfMemoryError, or the CPU
;; allocator's enforce-path failure) raise the typed exn:fail:rktorch:oom
;; instead of a plain exn:fail, so callers catch OOM by TYPE — never by
;; regexing messages. By the time a raise happens here, the allocator
;; layer's collect-and-retry (raw/memory.rkt) has already run for eligible
;; ops: an OOM surfacing to user code means one GC did not free enough.

(require (only-in "raw/global.rkt" tr-last-error/raw)
         (only-in "raw/memory.rkt" tr-last-error-kind/raw))

(provide check-ok
         check-handle
         (struct-out exn:fail:rktorch:oom))

(struct exn:fail:rktorch:oom exn:fail ())

;; Raise the failure the C side just recorded: typed when the paired kind
;; says OOM, the plain exn:fail `error` shape otherwise (message format
;; unchanged either way).
(define (raise-torch-failure who message)
  (define full (format "~a: ~a" who message))
  (if (= 1 (tr-last-error-kind/raw))
      (raise (exn:fail:rktorch:oom full (current-continuation-marks)))
      (raise (exn:fail full (current-continuation-marks)))))

(define (check-ok rc who)
  (unless (zero? rc)
    (raise-torch-failure
     who
     (format "FFI call failed (rc=~a): ~a" rc (tr-last-error/raw)))))

(define (check-handle who h)
  (unless h
    (raise-torch-failure
     who
     (format "torchrkt returned NULL handle: ~a" (tr-last-error/raw))))
  h)
