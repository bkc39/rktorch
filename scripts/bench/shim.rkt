#lang racket/base

(require (only-in ffi/unsafe ffi-lib))

(provide shim-library)

(define shim-library
  (let-values ([(dir _name _dir?) (split-path (collection-file-path "main.rkt" "torch"))])
    (ffi-lib (build-path dir "native-libs" "libtorchrkt"))))
