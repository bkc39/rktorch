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

(define (staged?)
  (and (directory-exists? native-libs-dir)
       (for/or ([f (in-list (directory-list native-libs-dir))])
         (regexp-match? #rx"^libtorchrkt[.]" (path->string f)))
       #t))

;; ffi-lib reports a missing file and a dlopen that failed on a transitive
;; dependency the same way, so the directory is probed separately.
(define (native-library-error present? where loader-message)
  (if present?
      (string-append
       "the native library libtorchrkt is staged but would not load.\n"
       "  found in: " (path->string where) "\n"
       "  The loader said: " loader-message "\n"
       "  A hand-copied library has to reach libtorch too; the one Nix builds"
       " carries an rpath to it.\n"
       "  See docs/building.md.")
      (string-append
       "the native library libtorchrkt is not staged.\n"
       "  looked in: " (path->string where) "\n"
       "  Build it with Nix -- `nix build`, or `nix develop`, which stages it"
       " for you --\n"
       "  or set TORCHRKT_NATIVE_LIB_PATH to a directory whose lib/ holds it"
       " and reinstall.\n"
       "  See docs/building.md.")))

(define native-library
  (with-handlers ([exn:fail?
                   (lambda (e)
                     (error 'torch "~a"
                            (native-library-error (staged?)
                                                  (simplify-path native-libs-dir)
                                                  (exn-message e))))])
    (ffi-lib (build-path native-libs-dir "libtorchrkt"))))

(define-ffi-definer define-torch native-library)

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

(module+ test
  (require (only-in rackunit check-false check-true test-case))

  (test-case "the two load failures do not read the same"
    (define absent
      (native-library-error #f (string->path "/x/native-libs") "ignored"))
    (define broken
      (native-library-error #t (string->path "/x/native-libs")
                            "libtorch.so: cannot open shared object file"))
    (check-true (regexp-match? #rx"is not staged" absent))
    (check-true (regexp-match? #rx"looked in: /x/native-libs" absent))
    (check-false (regexp-match? #rx"would not load" absent))
    (check-true (regexp-match? #rx"would not load" broken))
    (check-true (regexp-match? #rx"cannot open shared object file" broken)
                "the loader's own words survive")
    (check-false (regexp-match? #rx"is not staged" broken)))

  (test-case "a directory holding the library reads as staged"
    (check-true (staged?) "the dev shell stages it, and these tests need it")))
