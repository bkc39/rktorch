#lang racket/base

(provide pre-installer)

(require (only-in racket/file file->bytes make-directory*)
         (only-in "../foreign/raw/native-location.rkt"
                  native-library-directory
                  native-library-not-found-message)
         (only-in "util.rkt" with-temporary-file))

(define lib-pattern #rx"^libtorchrkt\\.")

;; rename(2), never a write in place: an in-place write faults every process
;; that has the old file mapped.
(define (copy-native-libs! dest-dir source-dir)
  (make-directory* dest-dir)
  (for ([f (in-list (directory-list source-dir))]
        #:when (regexp-match? lib-pattern (path->string f)))
    (define src (build-path source-dir f))
    (define dst (build-path dest-dir f))
    (cond
      [(and (file-exists? dst) (equal? (file->bytes src) (file->bytes dst)))
       (file-or-directory-permissions
        dst (file-or-directory-permissions src 'bits))]
      [else
        (with-temporary-file (tmp #:template "libtorchrkt-~a.part"
                                 #:directory dest-dir)
         (copy-file src tmp #t)
         (file-or-directory-permissions
          tmp (file-or-directory-permissions src 'bits))
         (rename-file-or-directory tmp dst #t))])))

(define (pre-installer _collections-top-path this-collection-path _user-specific?)
  (define native-libs-dir (build-path this-collection-path "native-libs"))
  (define cpp-lib-path (getenv "TORCHRKT_NATIVE_LIB_PATH"))
  (cond
    [cpp-lib-path
     (copy-native-libs! native-libs-dir (build-path cpp-lib-path "lib"))]
    [(native-library-directory #:staged native-libs-dir)
     (void)]
    [else
     (eprintf "torch: ~a\n"
              (native-library-not-found-message #:staged native-libs-dir))]))
