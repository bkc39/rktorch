#lang racket/base

;; Native library handle and FFI definer shared by every raw module.
;;
;; `native-libs-dir` is resolved relative to this file, which lives at
;; torchrkt/foreign/raw/ — two directories below the collection root — so the
;; path to torchrkt/native-libs/ climbs two levels.

(require ffi/unsafe
         ffi/unsafe/define
         racket/runtime-path)

(provide define-torchrkt)

(define-runtime-path native-libs-dir "../../native-libs")

(define-ffi-definer define-torchrkt
  (ffi-lib (build-path native-libs-dir "libtorchrkt")))
