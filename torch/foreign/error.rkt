#lang racket/base

;; Shared error helpers for the safe layer.  torchrkt C functions either return
;; an integer status (0 ok) with `tr_last_error` holding the message, or a
;; handle that is NULL on failure.

(require (only-in "raw/global.rkt" tr-last-error/raw))

(provide check-ok
         check-handle)

(define (check-ok rc who)
  (unless (zero? rc)
    (error who "FFI call failed (rc=~a): ~a" rc (tr-last-error/raw))))

(define (check-handle who h)
  (unless h
    (error who "torchrkt returned NULL handle: ~a" (tr-last-error/raw)))
  h)
