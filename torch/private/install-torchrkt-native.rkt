#lang racket/base

;; Pre-install hook: stage libtorchrkt.{so,dylib} into torch/native-libs/ so
;; `define-runtime-path` in foreign/raw/syntax.rkt resolves it.
;;
;; v0 supports the Nix path: the flake's racket build (and the `nix develop`
;; shell) set TORCHRKT_NATIVE_LIB_PATH to the cpp derivation, whose lib/ holds
;; the shared library; libtorch itself is found via the rpath baked into
;; libtorchrkt by Nix.  Catalog-style installs without Nix must pre-populate
;; native-libs/ (a portable-candidate story, as in xgboost-rkt, is deferred).

(provide pre-installer)

(require (only-in racket/file file->bytes make-directory*)
         (only-in "util.rkt" with-temporary-file))

(define lib-pattern #rx"^libtorchrkt\\.")

;; Temp file + rename(2), never a write in place: `cp`-style staging opens the
;; destination O_TRUNC, which invalidates the pages of any process still
;; executing the old file and faults it (#72).  rename swaps the directory
;; entry and leaves the old inode alive, so a live REPL keeps running the old
;; lib rather than crashing, and there is no window where the file is missing
;; or half-written.  Same discipline as data/mnist.rkt's cache write.
(define (copy-native-libs! dest-dir source-dir)
  (make-directory* dest-dir)
  (for ([f (in-list (directory-list source-dir))]
        #:when (regexp-match? lib-pattern (path->string f)))
    (define src (build-path source-dir f))
    (define dst (build-path dest-dir f))
    ;; Skip when the staged bytes already match: `raco setup` runs this on
    ;; every shell provision, right after the shell hook has staged the same
    ;; shim, and a redundant rename would churn the inode for nothing.
    (cond
      [(and (file-exists? dst) (equal? (file->bytes src) (file->bytes dst)))
       ;; Bytes match, but an older staging may have left a different mode.
       (file-or-directory-permissions
        dst (file-or-directory-permissions src 'bits))]
      [else
        (with-temporary-file (tmp #:template "libtorchrkt-~a.part"
                                 #:directory dest-dir)
         (copy-file src tmp #t)
         (file-or-directory-permissions
          tmp (file-or-directory-permissions src 'bits))
         (rename-file-or-directory tmp dst #t))])))

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
