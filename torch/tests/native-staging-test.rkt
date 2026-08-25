#lang racket/base

;; Staging must replace the directory entry, never write the file in place: a
;; process with the old library mapped keeps executing it, so the old inode
;; has to stay intact and readable across a restage.

(module+ test
  (require rackunit
           (only-in racket/file file->string make-directory* make-temporary-directory)
           (only-in racket/port port->string)
           (only-in "../private/install-torchrkt-native.rkt" pre-installer))

  ;; Build a fake `${cpp}` output whose lib/ holds a libtorchrkt with `content`.
  (define (make-src! content)
    (define root (make-temporary-directory))
    (define lib (build-path root "lib"))
    (make-directory lib)
    (call-with-output-file (build-path lib "libtorchrkt.so")
      (lambda (o) (write-string content o)))
    root)

  (define (stage! src collection)
    (parameterize ([current-environment-variables
                    (environment-variables-copy (current-environment-variables))])
      (putenv "TORCHRKT_NATIVE_LIB_PATH" (path->string src))
      (pre-installer #f collection #f)))

  (define (staged-path collection)
    (build-path collection "native-libs" "libtorchrkt.so"))

  (test-case "staging places the source bytes at the destination"
    (define collection (make-temporary-directory))
    (stage! (make-src! "SHIM-ONE") collection)
    (check-true (file-exists? (staged-path collection)))
    (check-equal? (file->string (staged-path collection)) "SHIM-ONE"))

  (test-case "re-staging replaces the inode instead of writing in place (#72)"
    ;; The property that saves a live REPL: after a restage, a handle opened
    ;; against the OLD file still reads the OLD bytes, because rename(2) only
    ;; swapped the directory entry.  An in-place write would fail this.
    (define collection (make-temporary-directory))
    (stage! (make-src! "OLD-SHIM") collection)
    (define dst (staged-path collection))
    (define id-before (file-or-directory-identity dst))
    (define held (open-input-file dst))
    (stage! (make-src! "NEW-SHIM") collection)
    (check-equal? (file->string dst) "NEW-SHIM" "new readers see the new shim")
    (check-not-equal? (file-or-directory-identity dst) id-before
                      "destination must be a NEW inode, not an overwritten one")
    (check-equal? (port->string held) "OLD-SHIM"
                  "a handle held across the restage must still see the old bytes")
    (close-input-port held))

  (test-case "staging leaves no temp file behind"
    (define collection (make-temporary-directory))
    (stage! (make-src! "SHIM") collection)
    (define leftovers
      (for/list ([f (in-list (directory-list (build-path collection "native-libs")))]
                 #:when (regexp-match? #rx"part|tmp" (path->string f)))
        f))
    (check-equal? leftovers '() "a .part/.tmp file survived staging"))

  (test-case "a failed copy leaves the previous shim intact and no temp file"
    (define collection (make-temporary-directory))
    (stage! (make-src! "GOOD-SHIM") collection)
    ;; source whose libtorchrkt.* entry is a directory: copy-file must fail
    (define bad (make-temporary-directory))
    (make-directory* (build-path bad "lib" "libtorchrkt.so"))
    (check-exn exn:fail? (lambda () (stage! bad collection)))
    (check-equal? (file->string (staged-path collection)) "GOOD-SHIM"
                  "a failed stage must not damage the installed shim")
    (check-equal?
     (for/list ([f (in-list (directory-list (build-path collection "native-libs")))]
                #:when (regexp-match? #rx"part|tmp" (path->string f)))
       f)
     '()
     "a .part/.tmp file survived a failed stage")))
