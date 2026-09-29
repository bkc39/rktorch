#lang racket/base

(require (only-in file/sha1 bytes->hex-string)
         (only-in net/url call/input-url get-pure-port string->url)
         (only-in racket/file make-directory*)
         (only-in racket/port copy-port make-limited-input-port)
         (only-in "util.rkt" with-temporary-file))

(provide call-with-verified-download
         download-cached)

;; the temporary file shares the cache's directory so the install's rename
;; stays on one filesystem and is atomic
(define (call-with-download url dir proc
                            #:headers headers
                            #:limit [limit #f])
  (make-directory* dir)
  (with-temporary-file (tmp #:template "download-~a.part" #:directory dir)
    (call/input-url (string->url url)
                    (lambda (u) (get-pure-port u headers #:redirections 5))
                    (lambda (in)
                      (call-with-output-file tmp #:exists 'truncate
                        (lambda (out)
                          (copy-port (if limit
                                         (make-limited-input-port in limit #f)
                                         in)
                                     out)))))
    (proc tmp)))

(define (call-with-verified-download who url dir size sha256 install
                                     #:headers [headers '()]
                                     #:limit [limit #f])
  (call-with-download
   url dir #:headers headers #:limit limit
   (lambda (tmp)
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
     (install tmp))))

(define (download-cached who dir name url
                         #:headers [headers '()]
                         #:valid? [valid? (lambda (_path) #t)])
  (define dest (build-path dir name))
  (unless (file-exists? dest)
    (define-values (parent _name _dir?) (split-path dest))
    (call-with-download
     url parent #:headers headers
     (lambda (tmp)
       (unless (valid? tmp)
         (raise (exn:fail:network
                 (format "~a: fetched ~a failed validation; not caching (bad response from ~a?)"
                         who name url)
                 (current-continuation-marks))))
       (rename-file-or-directory tmp dest #t))))
  dest)
