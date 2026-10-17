#lang racket/base

(require (only-in racket/contract/base -> ->* or/c)
         (only-in racket/path path-only)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in "../../private/contract.rkt" define/checked-out))

(define-runtime-path staged-native-libs "../../native-libs")

(define native-collection "torch-native")

(define platform-packages
  (hash "x86_64-linux" "torch-x86_64-linux"
        "aarch64-macosx" "torch-aarch64-macosx"))

(define (shim-file-name)
  (string-append "libtorchrkt" (bytes->string/utf-8 (system-type 'so-suffix))))

(define (holds-shim? dir)
  (and (directory-exists? dir)
       (for/or ([f (in-list (directory-list dir))])
         (regexp-match? #rx"^libtorchrkt[.]" (path->string f)))))

(define/checked-out (native-library-directory #:staged [staged staged-native-libs])
  (->* () (#:staged path-string?) (or/c path? #f))
  (cond
    [(holds-shim? staged) (simplify-path staged)]
    [else
     (define shim (collection-file-path (shim-file-name) native-collection
                                        #:fail (lambda (_) #f)))
     (and shim (file-exists? shim) (path-only shim))]))

(define/checked-out (platform-package [subpath (system-library-subpath #f)])
  (->* () (path-for-some-system?) (or/c string? #f))
  (hash-ref platform-packages (path->string subpath) #f))

(define/checked-out (platform-advice command [subpath (system-library-subpath #f)])
  (->* (string?) (path-for-some-system?) string?)
  (define package (platform-package subpath))
  (if package
      (string-append "  For a catalog install: " command " " package "\n")
      (string-append "  " (path->string subpath)
                     " has no prebuilt platform package; build the shim from source.\n")))

(define/checked-out (native-library-not-found-message
                     #:staged [staged staged-native-libs]
                     #:platform [subpath (system-library-subpath #f)])
  (->* () (#:staged path-string? #:platform path-for-some-system?) string?)
  (string-append
   "the native library libtorchrkt was not found.\n"
   "  looked in: " (path->string (simplify-path staged)) "\n"
   "  and in: collection " native-collection "\n"
   (platform-advice "raco pkg install" subpath)
   "  In a checkout, build it with Nix -- `nix build`, or `nix develop`, which"
   " stages it for you --\n"
   "  or set TORCHRKT_NATIVE_LIB_PATH to a directory whose lib/ holds it"
   " and reinstall.\n"
   "  See docs/building.md."))

;; ffi-lib reports a missing file and a dlopen that failed on a transitive
;; dependency the same way, so the resolver probes the directory first.
(define/checked-out (native-library-load-failure-message where loader-message)
  (-> path? string? string?)
  (string-append
   "the native library libtorchrkt was found but would not load.\n"
   "  found in: " (path->string where) "\n"
   "  The loader said: " loader-message "\n"
   "  A hand-copied library has to reach libtorch too; the one Nix builds"
   " carries an rpath to it.\n"
   "  See docs/building.md."))

(define/checked-out (native-library-path)
  (-> path?)
  (define where (native-library-directory))
  (unless where
    (error 'torch "~a" (native-library-not-found-message)))
  (build-path where "libtorchrkt"))
