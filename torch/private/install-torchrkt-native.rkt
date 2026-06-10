#lang racket/base

;; Pre-install hook: stage libtorchrkt.{so,dylib} into torchrkt/native-libs/ so
;; `define-runtime-path` in foreign/raw/library.rkt resolves it.
;;
;; v0 supports the Nix path: the flake's racket build (and the `nix develop`
;; shell) set TORCHRKT_NATIVE_LIB_PATH to the cpp derivation, whose lib/ holds
;; the shared library; libtorch itself is found via the rpath baked into
;; libtorchrkt by Nix.  Catalog-style installs without Nix must pre-populate
;; native-libs/ (a portable-candidate story, as in xgboost-rkt, is deferred).

(provide pre-installer)

(require (only-in racket/file make-directory*))

(define lib-pattern #rx"^libtorchrkt\\.")

(define (copy-native-libs! dest-dir source-dir)
  (make-directory* dest-dir)
  (for ([f (in-list (directory-list source-dir))]
        #:when (regexp-match? lib-pattern (path->string f)))
    (define src (build-path source-dir f))
    (define dst (build-path dest-dir f))
    (when (file-exists? dst)
      (delete-file dst))
    (copy-file src dst)))

(define (has-matching-files? dir)
  (and (directory-exists? dir)
       (pair? (filter (lambda (f) (regexp-match? lib-pattern (path->string f)))
                      (directory-list dir)))))

(define (pre-installer _collections-top-path this-collection-path _user-specific?)
  (define native-libs-dir (build-path this-collection-path "native-libs"))
  (define cpp-lib-path (getenv "TORCHRKT_NATIVE_LIB_PATH"))
  (cond
    [cpp-lib-path
     (copy-native-libs! native-libs-dir (build-path cpp-lib-path "lib"))]
    [(has-matching-files? native-libs-dir)
     (void)]
    [else
     (error
      'pre-installer
      (string-append
       "libtorchrkt not found. Either:\n"
       "  1. Build with Nix: `nix build` (or `nix develop`), which sets\n"
       "     TORCHRKT_NATIVE_LIB_PATH and stages the library, or\n"
       "  2. Copy libtorchrkt.* manually into ~a")
      (path->string native-libs-dir))]))
