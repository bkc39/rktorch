#lang racket/base

;; whole-module on purpose: the expansion needs bindings only-in would strip
(require racket/runtime-path
         (only-in racket/file file->string make-directory*)
         (only-in racket/port copy-port)
         (only-in racket/set for/set set->list)
         (only-in racket/string string-replace string-trim)
         (only-in net/url call/input-url get-pure-port string->url)
         (only-in "../private/util.rkt" with-temporary-file)
         (only-in "../main.rkt" narrow reshape tensor tensor->list
                  tensor-shape tensor? to-dtype))

(provide strip-gutenberg-boilerplate
         text->vocab
         encode
         decode
         contiguous-blocks
         load-text-fixture
         download-text-cached
         load-heart-of-darkness)

(define start-marker #rx"\\*\\*\\* START OF THE PROJECT GUTENBERG EBOOK [^\n]*\\*\\*\\*")
(define end-marker #rx"\\*\\*\\* END OF THE PROJECT GUTENBERG EBOOK")

(define (strip-gutenberg-boilerplate text)
  (define start (regexp-match-positions start-marker text))
  (unless start
    (error 'strip-gutenberg-boilerplate
           "no `*** START OF THE PROJECT GUTENBERG EBOOK ***` marker; ~a"
           "refusing to use the raw file (it would train on the license text)"))
  (define end (regexp-match-positions end-marker text (cdar start)))
  (unless end
    (error 'strip-gutenberg-boilerplate
           "no `*** END OF THE PROJECT GUTENBERG EBOOK` marker; ~a"
           "refusing to use the raw file (it would train on the license text)"))
  (string-trim (substring text (cdar start) (caar end))))

(define (text->vocab text)
  (list->vector (sort (set->list (for/set ([c (in-string text)]) c)) char<?)))

(define (encode vocab str)
  (define char->id
    (for/hash ([c (in-vector vocab)] [i (in-naturals)])
      (values c i)))
  (define (id-of c)
    (hash-ref char->id c
              (lambda () (error 'encode "char ~v not in vocab" c))))
  (to-dtype (tensor (for/list ([c (in-string str)]) (id-of c))) 'int64))

(define (decode vocab ids)
  (define id-list
    (if (tensor? ids) (map inexact->exact (tensor->list ids)) ids))
  (list->string (for/list ([i (in-list id-list)]) (vector-ref vocab i))))

;; deterministic (no shuffle) so the PyTorch parity twin sees identical
;; batches; xs and ys are overlapping narrow views over ids — read-only
(define (contiguous-blocks ids block-size)
  (define n (car (tensor-shape ids)))
  (define b (quotient (- n 1) block-size))
  (when (zero? b)
    (error 'contiguous-blocks
           "text too short (~a tokens) for even one block of ~a"
           n block-size))
  (values (reshape (narrow ids 0 0 (* b block-size)) b block-size)
          (reshape (narrow ids 0 1 (* b block-size)) b block-size)))

(define-runtime-path text-fixture "fixtures/heart-of-darkness-excerpt.txt")

(define (load-text-fixture)
  (file->string text-fixture))

(define heart-of-darkness-url
  "https://www.gutenberg.org/cache/epub/219/pg219.txt")

;; Gutenberg rate-limits clients with a bare/bot User-Agent.
(define gutenberg-headers
  (list "User-Agent: rktorch/0.1 (+https://github.com/bkc39/rktorch)"))

(define (text-cache-dir)
  (define override (getenv "RKTORCH_TEXT_DIR"))
  (if (and override (not (string=? override "")))
      (string->path override)
      (build-path (find-system-path 'cache-dir) "rktorch" "text")))

;; valid? gates the cache write: a rate-limit page or truncated body must
;; not poison the cache
(define (download-text-cached name url #:valid? [valid? (lambda (text) #t)])
  (define dest (build-path (text-cache-dir) name))
  (unless (file-exists? dest)
    (make-directory* (text-cache-dir))
    ;; temp file + atomic rename: an interrupted fetch must not poison the cache
    (with-temporary-file (tmp #:template "text-~a.part"
                              #:directory (text-cache-dir))
      (call/input-url (string->url url)
                      (lambda (u)
                        (get-pure-port u gutenberg-headers #:redirections 3))
                      (lambda (in)
                        (call-with-output-file tmp #:exists 'truncate
                          (lambda (out) (copy-port in out)))
                        (unless (valid? (file->string tmp))
                          (raise (exn:fail:network
                                  (format
                                   "download-text-cached: fetched ~a failed validation; not caching (bad response from ~a?)"
                                   name url)
                                  (current-continuation-marks))))
                        (rename-file-or-directory tmp dest #t)))))
  (file->string dest))

(define (gutenberg-text? text)
  (and (regexp-match? start-marker text)
       (regexp-match? end-marker text)))

(define (load-heart-of-darkness)
  (strip-gutenberg-boilerplate
   (string-replace (download-text-cached "pg219.txt" heart-of-darkness-url
                                         #:valid? gutenberg-text?)
                   "\r\n" "\n")))
