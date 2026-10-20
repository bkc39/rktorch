#lang racket/base

;; Staging must replace the directory entry, never write in place: the old
;; inode has to survive for processes that already mapped it.

(module+ test
  (require (only-in racket/file delete-directory/files file->string make-directory*
                    make-temporary-directory)
           (only-in racket/port port->string)
           rackunit
           (only-in "../foreign/raw/native-location.rkt" platform-package shim-file-name)
           (only-in "../private/install-torchrkt-native.rkt" pre-installer))

  (define created '())

  (define (temp-dir!)
    (define d (make-temporary-directory))
    (set! created (cons d created))
    d)

  (define (make-src! content)
    (define root (temp-dir!))
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

  (define (hook-says collection #:collects [collects (temp-dir!)])
    (parameterize ([current-environment-variables
                    (environment-variables-copy
                     (current-environment-variables))]
                   [current-error-port (open-output-string)]
                   [current-library-collection-links (list #f)]
                   [current-library-collection-paths (list collects)])
      (environment-variables-set!
       (current-environment-variables) #"TORCHRKT_NATIVE_LIB_PATH" #f)
      (check-not-exn (lambda () (pre-installer #f collection #f)))
      (get-output-string (current-error-port))))

  (define (touch-shim! dir)
    (make-directory* dir)
    (call-with-output-file (build-path dir (shim-file-name)) void))

  (test-case "with no source and nothing staged, the hook reports and returns"
    (define collection (temp-dir!))
    (define said (hook-says collection))
    (check-regexp-match #rx"libtorchrkt was not found" said)
    (check-regexp-match (regexp-quote (or (platform-package) "has no prebuilt platform package"))
                        said)
    (check-regexp-match #rx"TORCHRKT_NATIVE_LIB_PATH" said)
    (check-false (file-exists? (staged-path collection))))

  (test-case "with the shim staged, the hook says nothing"
    (define collection (temp-dir!))
    (touch-shim! (build-path collection "native-libs"))
    (check-equal? (hook-says collection) ""))

  (test-case "with the platform package's collection holding the shim, the hook says nothing"
    (define collects (temp-dir!))
    (touch-shim! (build-path collects "torch-native"))
    (check-equal? (hook-says (temp-dir!) #:collects collects) ""))

  (test-case "staging places the source bytes at the destination"
    (define collection (temp-dir!))
    (stage! (make-src! "SHIM-ONE") collection)
    (check-true (file-exists? (staged-path collection)))
    (check-equal? (file->string (staged-path collection)) "SHIM-ONE"))

  (test-case "re-staging replaces the inode instead of writing in place (#72)"
    (define collection (temp-dir!))
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

  (test-case "identical bytes skip the rename but still normalise the mode"
    (define collection (temp-dir!))
    (define src (make-src! "SAME-SHIM"))
    (stage! src collection)
    (define dst (staged-path collection))
    (define id-before (file-or-directory-identity dst))
    (define src-bits
      (file-or-directory-permissions (build-path src "lib" "libtorchrkt.so") 'bits))
    (define wrong-bits (if (= src-bits #o600) #o644 #o600))
    (file-or-directory-permissions dst wrong-bits)
    (check-not-equal? wrong-bits src-bits "the test must start from a differing mode")
    (stage! src collection)
    (check-equal? (file-or-directory-identity dst) id-before
                  "a stage of identical bytes must not churn the inode")
    (check-equal? (file-or-directory-permissions dst 'bits) src-bits
                  "the mode must be corrected even when the bytes match")
    (check-equal? (file->string dst) "SAME-SHIM"))

  (test-case "staging leaves no temp file behind"
    (define collection (temp-dir!))
    (stage! (make-src! "SHIM") collection)
    (define leftovers
      (for/list ([f (in-list (directory-list (build-path collection "native-libs")))]
                 #:when (regexp-match? #rx"part|tmp" (path->string f)))
        f))
    (check-equal? leftovers '() "a .part/.tmp file survived staging"))

  (test-case "a stage that fails while writing leaves the shim and no temp file"
    ;; Different bytes, so the equal-bytes branch is passed and the write path
    ;; is actually entered before it fails.
    (define collection (temp-dir!))
    (stage! (make-src! "GOOD-SHIM") collection)
    (define dst (staged-path collection))
    (define before (file-or-directory-identity dst))
    (define dest-dir (build-path collection "native-libs"))
    (file-or-directory-permissions dest-dir #o500)
    (check-exn exn:fail?
               (lambda () (stage! (make-src! "REPLACEMENT-SHIM") collection)))
    (file-or-directory-permissions dest-dir #o700)
    (check-equal? (file->string dst) "GOOD-SHIM"
                  "a failed stage must not damage the installed shim")
    (check-equal? (file-or-directory-identity dst) before
                  "a failed stage must not replace the inode")
    (check-equal?
     (for/list ([f (in-list (directory-list dest-dir))]
                #:when (regexp-match? #rx"part|tmp" (path->string f)))
       f)
     '()
     "a .part/.tmp file survived a failed stage")))
