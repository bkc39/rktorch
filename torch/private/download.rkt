#lang racket/base

(require (only-in file/sha1 bytes->hex-string)
         (only-in net/url call/input-url get-pure-port string->url)
         (only-in racket/file make-directory*)
         (only-in racket/port copy-port)
         (only-in "util.rkt" with-temporary-file))

(provide call-with-verified-download)

;; the download lands in a temporary file beside the cache and reaches
;; `install` only when its size and SHA-256 match, so a redirect page or a
;; transfer cut short never does
(define (call-with-verified-download who url dir size sha256 install)
  (make-directory* dir)
  (with-temporary-file (tmp #:template "download-~a.part" #:directory dir)
    (call/input-url (string->url url)
                    (lambda (u) (get-pure-port u #:redirections 5))
                    (lambda (in)
                      (call-with-output-file tmp #:exists 'truncate
                        (lambda (out) (copy-port in out)))))
    (define got (file-size tmp))
    (define digest
      (call-with-input-file tmp
        (lambda (in) (bytes->hex-string (sha256-bytes in)))))
    (unless (and (= got size) (string=? digest sha256))
      (raise-arguments-error who "download does not match the published file"
                             "url" url
                             "bytes" got
                             "expected bytes" size
                             "sha256" digest
                             "expected sha256" sha256))
    (install tmp)))
