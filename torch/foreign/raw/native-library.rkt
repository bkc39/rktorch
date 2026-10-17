#lang racket/base

(require (only-in ffi/unsafe _cprocedure _string/utf-8 ffi-lib ffi-lib? get-ffi-obj)
         (only-in racket/contract/base -> ->* or/c) ;; noqa
         (only-in racket/path path-only)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in "../../private/contract.rkt" define/checked-out)
         (only-in "fault.rkt" latched)
         (only-in "native-location.rkt"
                  native-library-load-failure-message
                  native-library-path
                  platform-advice))

(provide native-library)

(define-runtime-path abi-stamp-file "abi-stamp.rktd")

(define/checked-out bindings-abi-stamp
  string?
  (call-with-input-file abi-stamp-file read))

(define (abbreviated stamp)
  (substring stamp 0 (min 12 (string-length stamp))))

(define/checked-out (abi-stamp-problem bindings shim where)
  (-> string? (or/c string? #f) path? (or/c string? #f))
  (and (not (equal? bindings shim))
       (string-append
        (if shim
            (string-append
             "the native library libtorchrkt does not match these bindings.\n"
             "  shim ABI stamp: " (abbreviated shim) "\n")
            (string-append
             "the native library libtorchrkt predates these bindings:"
             " it has no tr_abi_stamp.\n"))
        "  bindings ABI stamp: " (abbreviated bindings) "\n"
        "  found in: " (path->string where) "\n"
        (platform-advice "raco pkg update torch")
        "  In a checkout: re-enter the Nix dev shell, or run"
        " `nix run .#copy-native-libs`.\n"
        "  See docs/building.md.")))

(define (shim-abi-stamp lib)
  (define stamp
    (get-ffi-obj "tr_abi_stamp" lib (_cprocedure '() _string/utf-8) (lambda () #f)))
  (and stamp ((latched stamp))))

(define/checked-out (open-native-library file #:expect [expected bindings-abi-stamp])
  (->* [path?] [#:expect string?] ffi-lib?)
  (define where (path-only file))
  (define lib
    (with-handlers ([exn:fail?
                     (lambda (e)
                       (error 'torch "~a"
                              (native-library-load-failure-message
                               where (exn-message e))))])
      (ffi-lib file)))
  (cond
    [(abi-stamp-problem expected (shim-abi-stamp lib) where)
     => (lambda (problem) (error 'torch "~a" problem))]
    [else lib]))

(define native-library (open-native-library (native-library-path)))
