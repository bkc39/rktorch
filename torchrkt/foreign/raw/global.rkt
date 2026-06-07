#lang racket/base

;; Raw global torchrkt APIs: libtorch version, last-error, and RNG seeding.

(require ffi/unsafe
         "library.rkt")

(provide tr-version/raw
         tr-last-error/raw
         tr-manual-seed/raw)

(define-torchrkt tr-version/raw
  (_fun -> _string/utf-8)
  #:c-id tr_version)

(define-torchrkt tr-last-error/raw
  (_fun -> _string/utf-8)
  #:c-id tr_last_error)

(define-torchrkt tr-manual-seed/raw
  (_fun _uint64 -> _int)
  #:c-id tr_manual_seed)
