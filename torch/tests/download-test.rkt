#lang racket/base

(module+ test
  (require (only-in file/sha1 bytes->hex-string)
           (only-in net/url path->url url->string)
           (only-in racket/file file->bytes file->string)
           (only-in racket/string string-join)
           (only-in racket/tcp tcp-accept tcp-addresses tcp-close tcp-listen)
           (only-in rackunit
                    check-equal? check-exn check-true test-case)
           (only-in "../private/download.rkt"
                    call-with-verified-download download-cached)
           (only-in "../private/util.rkt" with-temporary-directory))

  (define (sha256-hex bs) (bytes->hex-string (sha256-bytes bs)))

  (define (refused-for-validation? e)
    (and (exn:fail:network? e)
         (regexp-match? #rx"^test: fetched [^ ]+ failed validation; not caching"
                        (exn-message e))))

  (with-temporary-directory (dir)
    (define source (build-path dir "source.bin"))
    (call-with-output-file source (lambda (out) (write-bytes #"0123456789" out)))
    (define url (url->string (path->url source)))
    (define missing-url (url->string (path->url (build-path dir "missing.bin"))))

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
                                                file->bytes)))
      (check-equal? (directory-list dir) (list (string->path "source.bin"))
                    "a refused download leaves no temporary file"))

    (test-case "a download its check admits is installed, then served from the cache"
      (define cache (build-path dir "admitted"))
      (define seen #f)
      (define installed
        (download-cached 'test cache "corpus.bin" url
                         #:valid? (lambda (path)
                                    (set! seen (file->bytes path))
                                    #t)))
      (check-equal? installed (build-path cache "corpus.bin"))
      (check-equal? seen #"0123456789" "the check sees the downloaded file")
      (check-equal? (file->bytes installed) #"0123456789")
      (check-equal? (directory-list cache) (list (string->path "corpus.bin"))
                    "nothing but the installed file is left")
      (check-equal? (download-cached 'test cache "corpus.bin" missing-url
                                     #:valid? (lambda (_path) #f))
                    installed
                    "a cached name is neither fetched nor checked again"))

    (test-case "a download its check refuses leaves nothing in the cache"
      (define cache (build-path dir "refused"))
      (check-exn refused-for-validation?
                 (lambda ()
                   (download-cached 'test cache "corpus.bin" url
                                    #:valid? (lambda (_path) #f))))
      (check-equal? (directory-list cache) '())
      (check-exn #rx"the check's own complaint"
                 (lambda ()
                   (download-cached 'test cache "corpus.bin" url
                                    #:valid? (lambda (_path)
                                               (error "the check's own complaint")))))
      (check-equal? (directory-list cache) '()
                    "a check that raises refuses the download as well"))

    (test-case "a nested name creates its directory and fetches beside it"
      (define cache (build-path dir "nested"))
      (define installed (download-cached 'test cache "a/b/corpus.bin" url))
      (check-equal? (file->bytes installed) #"0123456789")
      (check-equal? (directory-list (build-path cache "a" "b"))
                    (list (string->path "corpus.bin")))))

  (define (read-request-head in)
    (let loop ([lines '()])
      (define line (read-line in 'return-linefeed))
      (if (or (eof-object? line) (string=? line ""))
          (reverse lines)
          (loop (cons line lines)))))

  (define (call-with-request-echo proc)
    (define listener
      (with-handlers ([exn:fail:network? (lambda (_e) #f)])
        (tcp-listen 0 1 #t "127.0.0.1")))
    (cond
      [listener
       (define-values (_host port _remote-host _remote-port)
         (tcp-addresses listener #t))
       (define server
         (thread
          (lambda ()
            (define-values (in out) (tcp-accept listener))
            (define body
              (string->bytes/utf-8 (string-join (read-request-head in) "\n")))
            (fprintf out "HTTP/1.0 200 OK\r\nContent-Length: ~a\r\n\r\n"
                     (bytes-length body))
            (write-bytes body out)
            (close-output-port out)
            (close-input-port in))))
       (dynamic-wind
        void
        (lambda () (proc (format "http://127.0.0.1:~a/corpus.txt" port)))
        (lambda ()
          (kill-thread server)
          (tcp-close listener)))]
      [else
       (displayln "[download-test] skipped headers (no loopback listener)")]))

  (call-with-request-echo
   (lambda (echo-url)
     (test-case "the headers reach the server"
       (with-temporary-directory (cache)
         (define installed
           (download-cached 'test cache "corpus.txt" echo-url
                            #:headers (list "X-Rktorch-Test: passed")))
         (check-true (regexp-match? #rx"(?m:^X-Rktorch-Test: passed$)"
                                    (file->string installed))))))))
