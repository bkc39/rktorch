#lang racket/base

(module+ test
  (require (only-in file/sha1 bytes->hex-string)
           (only-in net/url path->url url->string)
           (only-in racket/file file->bytes)
           (only-in rackunit check-equal? check-exn test-case)
           (only-in "../private/download.rkt" call-with-verified-download)
           (only-in "../private/util.rkt" with-temporary-directory))

  (define (sha256-hex bs) (bytes->hex-string (sha256-bytes bs)))

  (with-temporary-directory (dir)
    (define source (build-path dir "source.bin"))
    (call-with-output-file source (lambda (out) (write-bytes #"0123456789" out)))
    (define url (url->string (path->url source)))

    (test-case "a limited fetch keeps the prefix of a source that sends it all"
      (define installed
        (call-with-verified-download 'test url dir 4 (sha256-hex #"0123")
                                     file->bytes
                                     #:headers (list "Range: bytes=0-3")
                                     #:limit 4))
      (check-equal? installed #"0123"))

    (test-case "without the limit the whole response is checked, and refused"
      (check-exn #rx"does not match the published file.*bytes: 10"
                 (lambda ()
                   (call-with-verified-download 'test url dir 4
                                                (sha256-hex #"0123")
                                                file->bytes))))))
