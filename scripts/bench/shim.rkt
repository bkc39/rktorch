#lang racket/base

(require (only-in ffi/unsafe ffi-lib))

(provide torch-dir
         shim-library)

(define torch-dir
  (let-values ([(dir _name _dir?) (split-path (collection-file-path "main.rkt" "torch"))])
    dir))

(define shim-library
  (ffi-lib (build-path torch-dir "native-libs" "libtorchrkt")))
