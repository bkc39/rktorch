#lang racket/base

;; Char-level text data layer for the GPT capstone, over the Project
;; Gutenberg corpus (Heart of Darkness, ebook #219 — public domain). The
;; committed prose fixture keeps tests and parity runs offline.

;; whole-module on purpose: define-runtime-path expands into phase-1 code
;; that needs bindings only-in would strip
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

;; --- Project Gutenberg boilerplate ---------------------------------------
;; Every PG plain-text file wraps the work in START/END marker lines, with a
;; header before and the full PG license after.
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

;; --- char-level vocab ----------------------------------------------------
;; A char's index in the sorted vocab vector is its token id.
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

;; The tensor read path floatifies, so ids are re-exactified.
(define (decode vocab ids)
  (define id-list
    (if (tensor? ids) (map inexact->exact (tensor->list ids)) ids))
  (list->string (for/list ([i (in-list id-list)]) (vector-ref vocab i))))

;; --- contiguous-block batching -------------------------------------------
;; Deterministic — no shuffling — so the PyTorch parity twin sees identical
;; batches. xs and ys are overlapping `narrow` *views* over ids' storage:
;; treat all three as read-only, or in-place writes leak targets into inputs
;; with no error raised.
(define (contiguous-blocks ids block-size)
  (define n (car (tensor-shape ids)))
  (define b (quotient (- n 1) block-size))
  (when (zero? b)
    (error 'contiguous-blocks
           "text too short (~a tokens) for even one block of ~a"
           n block-size))
  (values (reshape (narrow ids 0 0 (* b block-size)) b block-size)
          (reshape (narrow ids 0 1 (* b block-size)) b block-size)))

;; --- committed fixture (offline) -----------------------------------------
(define-runtime-path text-fixture "fixtures/heart-of-darkness-excerpt.txt")

;; The committed excerpt is prose only, CRLF already normalized to \n.
(define (load-text-fixture)
  (file->string text-fixture))

;; --- full corpus (download + cache) --------------------------------------
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

;; `valid?` gates the cache write: a server can "succeed" with a rate-limit
;; page or truncated body, and caching that would make file-exists? skip the
;; download forever. An invalid body raises exn:fail:network — an
;; environmental transport failure the live tests self-skip on, not a bug.
(define (download-text-cached name url #:valid? [valid? (lambda (text) #t)])
  (define dest (build-path (text-cache-dir) name))
  (unless (file-exists? dest)
    (make-directory* (text-cache-dir))
    ;; Temp file + atomic rename: an interrupted fetch must not leave a
    ;; partial file at `dest` that file-exists? would treat as a valid cache.
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

;; A complete PG file carries both boilerplate markers; a rate-limit page
;; has neither and a truncated body loses END.
(define (gutenberg-text? text)
  (and (regexp-match? start-marker text)
       (regexp-match? end-marker text)))

;; PG serves DOS line endings; stray \r chars would pollute the char vocab.
(define (load-heart-of-darkness)
  (strip-gutenberg-boilerplate
   (string-replace (download-text-cached "pg219.txt" heart-of-darkness-url
                                         #:valid? gutenberg-text?)
                   "\r\n" "\n")))
