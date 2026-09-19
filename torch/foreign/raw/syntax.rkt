#lang racket/base

(require (for-syntax racket/base
                     ;; whole-module require on purpose
                     syntax/parse/pre)
         (only-in ffi/unsafe define-cpointer-type ffi-lib)
         (only-in ffi/unsafe/define define-ffi-definer)
         ;; whole-module require on purpose (only-in breaks its expansion)
         racket/runtime-path)

(provide define-torch
         _Tensor
         _Tensor/null ;; noqa
         Tensor? ;; noqa
         define-arith)

(define-runtime-path native-libs-dir "../../native-libs")

;; Without #:fail this reports the platform loader's own miss, naming a path
;; nobody chose and no way forward. The package installs and its docs render
;; without the library (the catalog's build server has none), so this is where
;; a user without one finds out.
(define (no-native-library)
  (error 'torch
         (string-append
          "the native library libtorchrkt is not staged.\n"
          "  looked in: ~a\n"
          "  Build it with Nix -- `nix build`, or `nix develop`, which stages"
          " it for you --\n"
          "  or set TORCHRKT_NATIVE_LIB_PATH to a directory whose lib/ holds"
          " it and reinstall.\n"
          "  See docs/building.md.")
         (simplify-path native-libs-dir)))

(define-ffi-definer define-torch
  (ffi-lib (build-path native-libs-dir "libtorchrkt")
           #:fail no-native-library))

(define-cpointer-type _Tensor)

(define-syntax (define-arith stx)
  (syntax-parse stx
    [(_ name:id tensor-pred:expr tensor-op:expr base-op:expr
        unary-tensor:expr)
     #'(define (name . args)
         (cond
           [(andmap number? args) (apply base-op args)]
           [(null? (cdr args))
            (let ([a (car args)])
              (if (tensor-pred a) (unary-tensor a) (base-op a)))]
           [else
            (foldl (lambda (b acc)
                     (if (or (tensor-pred acc) (tensor-pred b))
                         (tensor-op acc b)
                         (base-op acc b)))
                   (car args)
                   (cdr args))]))]))
