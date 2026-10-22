#lang racket/base

(require (only-in ffi/unsafe _fun _int ffi-lib get-ffi-obj)
         (only-in racket/match match))

(provide torch-dir
         checkout-module
         shim-library
         aten-threads)

(define torch-dir
  (let-values ([(dir _name _dir?) (split-path (collection-file-path "main.rkt" "torch"))])
    dir))

(define (checkout-module #:torch [dir torch-dir] . segments)
  (define path (apply build-path dir segments))
  (and (file-exists? path) path))

(define shim-library
  (ffi-lib
   (match (checkout-module "foreign" "raw" "native-location.rkt")
     [#f (build-path torch-dir "native-libs" "libtorchrkt")]
     [m ((dynamic-require `(submod (file ,(path->string m)) checked) 'native-library-path))])))

(define aten-threads
  (get-ffi-obj "_ZN2at15get_num_threadsEv" shim-library (_fun -> _int)
               (lambda () (lambda () #f))))
