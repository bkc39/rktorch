#lang racket/base

(module+ test
  (require (only-in racket/file make-directory*)
           (only-in racket/string string-contains?)
           rackunit
           (only-in (submod "../foreign/raw/native-library.rkt" checked)
                    abi-stamp-problem bindings-abi-stamp open-native-library)
           (only-in (submod "../foreign/raw/native-location.rkt" checked)
                    native-library-directory native-library-load-failure-message
                    native-library-not-found-message native-library-path
                    platform-advice platform-package shim-file-name)
           (only-in "../private/util.rkt" with-temporary-directory))

  (define (touch! dir name)
    (make-directory* dir)
    (call-with-output-file (build-path dir name) void)
    dir)

  (define (call-with-layout proc)
    (with-temporary-directory (root)
      (define staged (build-path root "native-libs"))
      (define collects (build-path root "collects"))
      (make-directory* staged)
      (make-directory* collects)
      (parameterize ([current-library-collection-links (list #f)]
                     [current-library-collection-paths (list collects)])
        (proc staged (build-path collects "torch-native")))))

  (test-case "a staged shim wins over the platform package"
    (call-with-layout
     (lambda (staged native)
       (touch! staged (shim-file-name))
       (touch! native (shim-file-name))
       (check-equal? (native-library-directory #:staged staged)
                     (simplify-path staged)))))

  (test-case "with nothing staged, the platform package's collection supplies it"
    (call-with-layout
     (lambda (staged native)
       (touch! staged "README")
       (touch! native (shim-file-name))
       (check-equal? (native-library-directory #:staged staged)
                     (path->directory-path native)))))

  (test-case "a staged library under another suffix does not shadow the platform package"
    (call-with-layout
     (lambda (staged native)
       (touch! staged (if (equal? (system-type 'so-suffix) #".so")
                          "libtorchrkt.dylib"
                          "libtorchrkt.so"))
       (touch! staged (string-append (shim-file-name) ".1"))
       (touch! native (shim-file-name))
       (check-equal? (native-library-directory #:staged staged)
                     (path->directory-path native)))))

  (test-case "with neither, the path raises the not-found message"
    (call-with-layout
     (lambda (staged native)
       (make-directory* native)
       (check-exn #rx"was not found" (lambda () (native-library-path #:staged staged))))))

  (test-case "with neither, there is no directory"
    (call-with-layout
     (lambda (staged native)
       (make-directory* native)
       (check-false (native-library-directory #:staged staged))
       (check-false (native-library-directory #:staged (build-path staged "absent"))))))

  (test-case "the not-found message names both places and the host's package"
    (define said (native-library-not-found-message #:staged "/x/native-libs"))
    (check-regexp-match #rx"was not found" said)
    (check-regexp-match #rx"looked in: /x/native-libs" said)
    (check-regexp-match #rx"collection torch-native" said)
    (check-regexp-match (regexp-quote (or (platform-package) "has no prebuilt platform package"))
                        said)
    (check-regexp-match #rx"docs/building[.]md" said))

  (test-case "each prebuilt platform names its package; any other says there is none"
    (define (said platform)
      (native-library-not-found-message #:staged "/x" #:platform (string->path platform)))
    (check-regexp-match #rx"raco pkg install torch-x86_64-linux" (said "x86_64-linux"))
    (check-regexp-match #rx"raco pkg install torch-x86_64-linux" (said "x86_64-linux-natipkg"))
    (check-regexp-match #rx"raco pkg install torch-aarch64-macosx" (said "aarch64-macosx"))
    (check-regexp-match #rx"raco pkg install torch-aarch64-macosx" (said "aarch64-darwin"))
    (check-regexp-match #rx"riscv64-linux has no prebuilt platform package"
                        (said "riscv64-linux"))
    (check-false (regexp-match? #rx"raco pkg" (said "riscv64-linux"))))

  (test-case "a shim that was found but would not load reads differently"
    (define broken
      (native-library-load-failure-message
       (string->path "/x/native-libs") "libtorch.so: cannot open shared object file"))
    (check-regexp-match #rx"found but would not load" broken)
    (check-regexp-match #rx"found in: /x/native-libs" broken)
    (check-regexp-match #rx"cannot open shared object file" broken)
    (check-false (regexp-match? #rx"was not found" broken)))

  (test-case "a shim file that would not open raises the load failure"
    (with-temporary-directory (dir)
      (touch! dir (shim-file-name))
      (check-exn #rx"found but would not load"
                 (lambda () (open-native-library (build-path dir "libtorchrkt"))))))

  (define bindings (make-string 64 #\a))
  (define where (string->path "/x/native-libs"))

  (test-case "a matching stamp is no problem"
    (check-false (abi-stamp-problem bindings (string-copy bindings) where)))

  (test-case "a different stamp names both, where the shim is, and the fixes"
    (define said (abi-stamp-problem bindings (make-string 64 #\b) where))
    (check-regexp-match #rx"does not match these bindings" said)
    (check-regexp-match #rx"shim ABI stamp: bbbbbbbbbbbb" said)
    (check-regexp-match #rx"bindings ABI stamp: aaaaaaaaaaaa" said)
    (check-regexp-match #rx"found in: /x/native-libs" said)
    (check-regexp-match (regexp-quote (or (platform-package) "has no prebuilt platform package"))
                        said)
    (check-regexp-match #rx"nix run [.]#abi-stamp` and commit the stamp" said)
    (check-regexp-match #rx"nix run [.]#copy-native-libs" said))

  (test-case "a shim without the stamp symbol says it predates the stamp"
    (define said (abi-stamp-problem bindings #f where))
    (check-regexp-match #rx"no tr_abi_stamp" said)
    (check-regexp-match #rx"bindings ABI stamp: aaaaaaaaaaaa" said)
    (check-true (string-contains? said (platform-advice "raco pkg update torch"))))

  (test-case "the committed stamp is a SHA-256"
    (check-regexp-match #px"^[0-9a-f]{64}$" bindings-abi-stamp))

  (test-case "the real shim under another expected stamp raises the mismatch, not an FFI error"
    (check-exn (lambda (e)
                 (and (exn:fail? e)
                      (regexp-match? #rx"does not match these bindings" (exn-message e))
                      (regexp-match? (regexp-quote (substring bindings-abi-stamp 0 12))
                                     (exn-message e))))
               (lambda () (open-native-library (native-library-path) #:expect bindings))))

  (test-case "the real shim under its own stamp opens"
    (check-not-exn (lambda () (open-native-library (native-library-path))))))
