#lang racket/base

;; Raw global torchrkt APIs: libtorch version, last-error, and RNG seeding.

(require (only-in ffi/unsafe _fun _int _string/utf-8 _uint64)
         ;; the kind probe lives in memory.rkt (avoiding a require cycle);
         ;; re-provided here so this module carries the full last-error API
         (only-in "memory.rkt" tr-last-error-kind/raw)
         (only-in "syntax.rkt" define-torch))

(provide tr-version/raw
         tr-last-error/raw
         tr-last-error-kind/raw
         tr-manual-seed/raw)

(define-torch tr-version/raw
  (_fun -> _string/utf-8)
  #:c-id tr_version)

(define-torch tr-last-error/raw
  (_fun -> _string/utf-8)
  #:c-id tr_last_error)

(define-torch tr-manual-seed/raw
  (_fun _uint64 -> _int)
  #:c-id tr_manual_seed)
