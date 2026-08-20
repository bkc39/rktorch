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

(require (only-in ffi/unsafe/atomic call-as-atomic)
         (only-in "raw/global.rkt" tr-last-error-kind/raw tr-last-error/raw))

(provide check-ok
         check-handle
         (struct-out exn:fail:rktorch:oom))

(struct exn:fail:rktorch:oom exn:fail ())

;; Read the recorded (message, kind) pair without a green-thread
;; scheduling gap between the two FFI reads: the C side records them
;; together, and atomic mode keeps another thread's torchrkt failure
;; from landing between our reads — the pair we raise from is always
;; internally consistent. (The wider window between the FAILING call
;; and this read predates this PR and is catalogued in #40.)
(define (last-failure)
  (call-as-atomic
   (lambda () (values (tr-last-error/raw) (tr-last-error-kind/raw)))))

;; Raise the failure the C side just recorded: typed when the paired kind
;; says OOM, the plain exn:fail `error` shape otherwise (message format
;; unchanged either way). `describe` renders the caller's message around
;; the atomically-read error text.
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
