#lang racket/base

;; Native library handle and FFI definer shared by every raw module.
;;
;; `native-libs-dir` is resolved relative to this file, which lives at
;; torch/foreign/raw/ — two directories below the collection root — so the
;; path to torch/native-libs/ climbs two levels.

(require (only-in ffi/unsafe ffi-lib)
         (only-in ffi/unsafe/define define-ffi-definer)
         ;; whole-module on purpose: define-runtime-path expands into
         ;; phase-1 code that needs bindings (e.g. #%datum) the full require
         ;; re-exports for-syntax; only-in strips them.
         racket/runtime-path)

(provide define-torchrkt)

(define-runtime-path native-libs-dir "../../native-libs")

(define-ffi-definer define-torchrkt
  (ffi-lib (build-path native-libs-dir "libtorchrkt")))
