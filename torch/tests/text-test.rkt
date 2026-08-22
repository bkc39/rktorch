#lang racket/base

(module+ test
  (require rackunit
           (only-in racket/file
                    delete-directory/files
                    display-to-file
                    make-temporary-directory)
           (only-in racket/list take)
           (only-in racket/string string-contains?)
           (only-in "../main.rkt" tensor-shape tensor->list)
           (only-in "../data/text.rkt"
                    strip-gutenberg-boilerplate
                    text->vocab
                    encode
                    decode
                    contiguous-blocks
                    load-text-fixture
                    download-text-cached
                    load-heart-of-darkness))

  (test-case "strip-gutenberg-boilerplate cuts to the prose"
    (define wrapped
      (string-append
       "The Project Gutenberg eBook of X\nrelease notes\n\n"
       "*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\n"
       "prose line one\nprose line two\n\n"
       "*** END OF THE PROJECT GUTENBERG EBOOK X ***\n\nlicense text"))
    (check-equal? (strip-gutenberg-boilerplate wrapped)
                  "prose line one\nprose line two"))

  (test-case "strip-gutenberg-boilerplate errors loudly on missing markers"
    (check-exn #rx"no `\\*\\*\\* START"
               (lambda () (strip-gutenberg-boilerplate "no markers here")))
    (check-exn #rx"no `\\*\\*\\* END"
               (lambda ()
                 (strip-gutenberg-boilerplate
                  "*** START OF THE PROJECT GUTENBERG EBOOK X ***\nprose"))))

  (test-case "fixture is the CRLF-normalized opening prose"
    (define s (load-text-fixture))
    (check-equal? (string-length s) 841)
    (check-true (regexp-match? #rx"^The Nellie, a cruising yawl" s))
    (check-false (string-contains? s "\r") "fixture must be LF-only"))

  (test-case "vocab is sorted + unique; encode/decode round-trips"
    (define s (load-text-fixture))
    (define vocab (text->vocab s))
    (for ([a (in-vector vocab)] [b (in-vector vocab 1)])
      (check-true (char<? a b) "vocab must be strictly increasing"))
    (define ids (encode vocab s))
    (check-equal? (tensor-shape ids) (list (string-length s)))
    (check-equal? (decode vocab ids) s "encode/decode round-trip")
    (check-equal? (decode vocab '(0)) (string (vector-ref vocab 0))))

  (test-case "encode rejects a char outside the vocab"
    (define vocab (text->vocab "ab"))
    (check-exn #rx"not in vocab" (lambda () (encode vocab "abc"))))

  (test-case "contiguous-blocks: shapes and the one-char shift"
    (define s (load-text-fixture))
    (define vocab (text->vocab s))
    (define ids (encode vocab s))
    (define-values (xs ys) (contiguous-blocks ids 16))
    (check-equal? (tensor-shape xs) '(52 16))
    (check-equal? (tensor-shape ys) '(52 16))
    (define id-list (map inexact->exact (tensor->list ids)))
    (check-equal? (map inexact->exact (tensor->list xs))
                  (take id-list (* 52 16)))
    (check-equal? (map inexact->exact (tensor->list ys))
                  (take (cdr id-list) (* 52 16)))
    (check-exn #rx"too short"
               (lambda () (contiguous-blocks (encode vocab "ab") 16))))

  (test-case "download-text-cached: a failed validation must not poison the cache"
    (define scratch (make-temporary-directory "rktorch-text-test-~a"))
    (dynamic-wind
     void
     (lambda ()
       (parameterize ([current-environment-variables
                       (environment-variables-copy
                        (current-environment-variables))])
         (putenv "RKTORCH_TEXT_DIR" (path->string scratch))
         (define (file-url p) (string-append "file://" (path->string p)))
         (define (has-markers? s) (string-contains? s "***"))
         (define bad (build-path scratch "bad-source.txt"))
         (display-to-file "<html>429 Too Many Requests</html>" bad)
         (check-exn (lambda (e)
                      (and (exn:fail:network? e)
                           (regexp-match? #rx"failed validation"
                                          (exn-message e))))
                    (lambda ()
                      (download-text-cached "corpus.txt" (file-url bad)
                                            #:valid? has-markers?)))
         (check-false (file-exists? (build-path scratch "corpus.txt"))
                      "invalid body must not be cached")
         (define good (build-path scratch "good-source.txt"))
         (display-to-file "*** wrapped prose ***" good)
         (check-equal? (download-text-cached "corpus.txt" (file-url good)
                                             #:valid? has-markers?)
                       "*** wrapped prose ***")
         (delete-file good)
         (check-equal? (download-text-cached "corpus.txt" (file-url good)
                                             #:valid? has-markers?)
                       "*** wrapped prose ***"
                       "cache hit must not refetch")))
     (lambda () (delete-directory/files scratch))))

  ;; Only environmental failures (network/filesystem) may skip; a stripper
  ;; regression raises plain exn:fail and fails.
  (define prose
    (with-handlers ([exn:fail:network? (lambda (_) #f)]
                    [exn:fail:filesystem? (lambda (_) #f)])
      (load-heart-of-darkness)))
  (cond
    [prose
     (check-true (regexp-match? #rx"^Heart of Darkness" prose)
                 "prose starts at the title page (PG header stripped)")
     (check-true (string-contains? prose "The Nellie, a cruising yawl")
                 "novella text present")
     (check-false (string-contains? prose "PROJECT GUTENBERG")
                  "PG license/end matter stripped")
     (check-true (> (string-length prose) 100000) "full corpus length")
     (displayln "[text-test] load-heart-of-darkness OK (prose stripped)")]
    [else
     (displayln
      "[text-test] skipped download (offline / Gutenberg unreachable)")]))
