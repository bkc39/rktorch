#lang racket/base

(require (only-in ffi/unsafe _fun _int ffi-lib get-ffi-obj))

(provide torch-dir
         shim-library
         aten-threads)

(define torch-dir
  (let-values ([(dir _name _dir?) (split-path (collection-file-path "main.rkt" "torch"))])
    dir))

(define shim-library
  (ffi-lib (build-path torch-dir "native-libs" "libtorchrkt")))

(define aten-threads
  (get-ffi-obj "_ZN2at15get_num_threadsEv" shim-library (_fun -> _int)
               (lambda () (lambda () #f))))
