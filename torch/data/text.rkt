#lang racket/base

;; Char-level text data layer for the GPT capstone: a download-with-cache
;; loader for the Project Gutenberg corpus (Joseph Conrad, Heart of Darkness,
;; ebook #219 — public domain), PG boilerplate stripping, a committed prose
;; fixture so tests and parity runs stay offline, and the vocab / encode /
;; decode / contiguous-block helpers a char-LM trains on.

;; whole-module on purpose: define-runtime-path expands into phase-1 code
;; that needs bindings (#%datum, ...) only-in would strip (the documented
;; runtime-path exemption to the only-in convention).
(require racket/runtime-path
         (only-in racket/file file->string make-directory*)
         (only-in racket/port copy-port)
         (only-in racket/set for/set set->list)
         (only-in racket/string string-replace string-trim)
         (only-in net/url string->url get-pure-port call/input-url)
         (only-in "../private/util.rkt" with-temporary-file)
         (only-in "../main.rkt" tensor tensor? tensor-shape tensor->list
                  to-dtype narrow reshape))

(provide strip-gutenberg-boilerplate
         text->vocab
         encode
         decode
         contiguous-blocks
         load-text-fixture
         download-text-cached
         load-heart-of-darkness)

;; --- Project Gutenberg boilerplate ---------------------------------------
;; Every PG plain-text file wraps the work in `*** START/END OF THE PROJECT
;; GUTENBERG EBOOK <TITLE> ***` marker lines, with a header before and the
;; full PG license after.
(define start-marker #rx"\\*\\*\\* START OF THE PROJECT GUTENBERG EBOOK [^\n]*\\*\\*\\*")
(define end-marker #rx"\\*\\*\\* END OF THE PROJECT GUTENBERG EBOOK")

;; Cut `text` down to the prose between the PG markers (whitespace-trimmed).
;; Errors loudly when a marker is missing: silently returning the raw file
;; would put the PG header/license in the training data.
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
;; The sorted unique characters of `text`, as a vector; a char's index in
;; the vector is its token id.
(define (text->vocab text)
  (list->vector (sort (set->list (for/set ([c (in-string text)]) c)) char<?)))

;; Encode `str` as an [N] int64 token tensor; errors on a char outside the
;; vocab (train/sample text must come from the same corpus as the vocab).
(define (encode vocab str)
  (define char->id
    (for/hash ([c (in-vector vocab)] [i (in-naturals)])
      (values c i)))
  (define (id-of c)
    (hash-ref char->id c
              (lambda () (error 'encode "char ~v not in vocab" c))))
  (to-dtype (tensor (for/list ([c (in-string str)]) (id-of c))) 'int64))

;; Decode token ids (an int64 tensor, or a list of naturals) back to a
;; string. The tensor read path floatifies, so ids are re-exactified.
(define (decode vocab ids)
  (define id-list
    (if (tensor? ids) (map inexact->exact (tensor->list ids)) ids))
  (list->string (for/list ([i (in-list id-list)]) (vector-ref vocab i))))

;; --- contiguous-block batching -------------------------------------------
;; Split `ids` (an [N] int64 token tensor) into next-char training pairs:
;; (values xs ys), both [B, block-size] int64 with B = floor((N-1)/block-size),
;; where row b of xs is the b-th contiguous block of the text and ys is the
;; same block shifted one char ahead (the LM target). Deterministic — no
;; shuffling — so the PyTorch parity twin sees identical batches. The last
;; partial block (and its +1 target lookahead) is dropped.
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

;; The committed excerpt (the novella's first two paragraphs, cut from the
;; real pg219.txt): prose only, CRLF already normalized to \n.
(define (load-text-fixture)
  (file->string text-fixture))

;; --- full corpus (download + cache) --------------------------------------
(define heart-of-darkness-url
  "https://www.gutenberg.org/cache/epub/219/pg219.txt")

;; Gutenberg rate-limits clients with a bare/bot User-Agent; identify
;; ourselves like a normal client.
(define gutenberg-headers
  (list "User-Agent: rktorch/0.1 (+https://github.com/bkc39/rktorch)"))

;; Cache location: $RKTORCH_TEXT_DIR if set (e.g. a big data disk), else the
;; per-user system cache dir.
(define (text-cache-dir)
  (define override (getenv "RKTORCH_TEXT_DIR"))
  (if (and override (not (string=? override "")))
      (string->path override)
      (build-path (find-system-path 'cache-dir) "rktorch" "text")))

;; Fetch `url` into the cache once (as `name`), then return the file's
;; contents as a UTF-8 string.
(define (download-text-cached name url)
  (define dest (build-path (text-cache-dir) name))
  (unless (file-exists? dest)
    (make-directory* (text-cache-dir))
    ;; Download to a temp file in the cache dir, then atomically rename on
    ;; success, so an interrupted/failed fetch can't leave a partial file at
    ;; `dest` that file-exists? would treat as a valid cache entry.
    ;; with-temporary-file owns the temp's lifetime (removed on any escape) and
    ;; call/input-url owns the URL port (closed even on error) — so neither the
    ;; temp nor the port leaks on a network failure. Same dance as mnist.rkt's
    ;; download-cached, plus the User-Agent header and redirect allowance.
    (with-temporary-file (tmp #:template "text-~a.part"
                              #:directory (text-cache-dir))
      (call/input-url (string->url url)
                      (lambda (u)
                        (get-pure-port u gutenberg-headers #:redirections 3))
                      (lambda (in)
                        (call-with-output-file tmp #:exists 'truncate
                          (lambda (out) (copy-port in out)))
                        (rename-file-or-directory tmp dest #t)))))
  (file->string dest))

;; The full Heart of Darkness prose: downloaded (cached), CRLF normalized to
;; \n (PG serves DOS line endings; \r chars would pollute the char vocab),
;; PG boilerplate stripped.
(define (load-heart-of-darkness)
  (strip-gutenberg-boilerplate
   (string-replace (download-text-cached "pg219.txt" heart-of-darkness-url)
                   "\r\n" "\n")))
