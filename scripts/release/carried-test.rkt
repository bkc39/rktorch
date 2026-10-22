#lang racket/base
(require (only-in file/sha1 bytes->hex-string sha1)
         (only-in json write-json)
         (only-in racket/file display-to-file make-directory*)
         (only-in racket/port with-output-to-string)
         (only-in racket/pretty pretty-write)
         (only-in rackunit check-equal? test-case)
         (only-in "../../torch/private/util.rkt" with-temporary-directory)
         (only-in "carried.rkt" carried-source-urls check-carried)
         (only-in "fixture.rkt" refused)
         (only-in "release.rkt" run))

(define zip-bytes #"pretend platform zip\n")
(define zip-sha256 (bytes->hex-string (sha256-bytes zip-bytes)))
(define zip-sha1 (sha1 (open-input-bytes zip-bytes)))
(define name "torch-x86_64-linux")

(define (manifest #:sha256 [sha256 zip-sha256])
  (hasheq 'natives-tag "v0.0.1-alpha"
          'platform-packages (list (hasheq 'name "torch-aarch64-macosx" 'sha256 "other")
                                   (hasheq 'name name 'sha256 sha256))
          'sources (list (hasheq 'file "lame-3.100.tar.gz")
                         (hasheq 'file "mpg123-1.33.4.tar.bz2"))))

(define (catalog #:checksum [checksum zip-sha1])
  (hash name (hash 'name name 'checksum checksum)))

(define (write-zip dir #:checksum [checksum zip-sha1])
  (define zip (build-path dir (string-append name ".zip")))
  (display-to-file zip-bytes zip #:exists 'truncate)
  (display-to-file (string-append checksum "\n") (format "~a.CHECKSUM" zip) #:exists 'truncate)
  zip)

(test-case "a carried zip is the one the previous release built, by sha256 and by .CHECKSUM"
  (with-temporary-directory (dir)
    (define zip (write-zip dir))
    (check-equal? (check-carried (manifest) (catalog) name zip) (void))
    (refused #rx"torch-x86_64-linux.zip has sha256 [0-9a-f]+, but the previous manifest.json released 0+$"
             (λ () (check-carried (manifest #:sha256 (make-string 64 #\0)) (catalog) name zip)))
    (refused #rx"CHECKSUM holds [0-9a-f]+, but the previous catalog-entry.rktd's torch-x86_64-linux checksum is 0+$"
             (λ () (check-carried (manifest) (catalog #:checksum (make-string 40 #\0)) name zip)))
    (write-zip dir #:checksum (make-string 40 #\1))
    (refused #rx"CHECKSUM holds 1+, but"
             (λ () (check-carried (manifest) (catalog) name zip)))))

(test-case "a carried package missing from the previous manifest or catalog fails"
  (with-temporary-directory (dir)
    (define zip (write-zip dir))
    (refused #rx"no platform package torch-x86_64-linux"
             (λ () (check-carried (hash-set (manifest) 'platform-packages '()) (catalog) name zip)))
    (refused #rx"no platform-packages list"
             (λ () (check-carried (hasheq) (catalog) name zip)))
    (refused #rx"torch-x86_64-linux entry has no sha256"
             (λ () (check-carried (manifest #:sha256 #f) (catalog) name zip)))
    (refused #rx"catalog-entry.rktd has no checksum for torch-x86_64-linux"
             (λ () (check-carried (manifest) (hash) name zip)))))

(test-case "the carried sources are the assets beside the natives' release"
  (check-equal? (carried-source-urls (manifest) "owner/repo")
                '("https://github.com/owner/repo/releases/download/v0.0.1-alpha/lame-3.100.tar.gz"
                  "https://github.com/owner/repo/releases/download/v0.0.1-alpha/mpg123-1.33.4.tar.bz2"))
  (refused #rx"needs a natives-tag string and a sources list"
           (λ () (carried-source-urls (hasheq 'natives-tag "v1") "owner/repo")))
  (refused #rx"a source in the previous manifest.json has no file"
           (λ () (carried-source-urls (hash-set (manifest) 'sources (list (hasheq))) "owner/repo"))))

(test-case "release.rkt check-carried and carried-sources read the previous release's files"
  (with-temporary-directory (dir)
    (define previous (build-path dir "previous"))
    (make-directory* previous)
    (with-output-to-file (build-path previous "manifest.json")
      (λ () (write-json (manifest))))
    (with-output-to-file (build-path previous "catalog-entry.rktd")
      (λ () (pretty-write (catalog))))
    (define zip (write-zip dir))
    (check-equal? (with-output-to-string
                    (λ () (run (list "check-carried" (path->string previous) name (path->string zip)))))
                  "")
    (refused #rx"previous catalog-entry.rktd's torch-x86_64-linux checksum"
             (λ () (run (list "check-carried" (path->string previous) name
                              (path->string (write-zip dir #:checksum (make-string 40 #\2)))))))
    (check-equal? (with-output-to-string
                    (λ () (run (list "carried-sources" (path->string (build-path previous "manifest.json"))
                                     "owner/repo"))))
                  (string-append
                   "https://github.com/owner/repo/releases/download/v0.0.1-alpha/lame-3.100.tar.gz\n"
                   "https://github.com/owner/repo/releases/download/v0.0.1-alpha/mpg123-1.33.4.tar.bz2\n"))))
