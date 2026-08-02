#lang racket/base

;; Tests for the char-level text data layer. The committed-fixture cases run
;; offline (and in the sandboxed nix build); the full-download case is
;; network-guarded and self-skips when Gutenberg is unreachable, like
;; mnist-test.

(module+ test
  (require rackunit
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
    ;; strictly increasing == sorted with no duplicates
    (for ([a (in-vector vocab)] [b (in-vector vocab 1)])
      (check-true (char<? a b) "vocab must be strictly increasing"))
    (define ids (encode vocab s))
    (check-equal? (tensor-shape ids) (list (string-length s)))
    (check-equal? (decode vocab ids) s "encode/decode round-trip")
    ;; decode also accepts a plain id list (the generation loop's case)
    (check-equal? (decode vocab '(0)) (string (vector-ref vocab 0))))

  (test-case "encode rejects a char outside the vocab"
    (define vocab (text->vocab "ab"))
    (check-exn #rx"not in vocab" (lambda () (encode vocab "abc"))))

  (test-case "contiguous-blocks: shapes and the one-char shift"
    (define s (load-text-fixture))
    (define vocab (text->vocab s))
    (define ids (encode vocab s))
    (define-values (xs ys) (contiguous-blocks ids 16))
    ;; 841 chars -> floor(840/16) = 52 blocks
    (check-equal? (tensor-shape xs) '(52 16))
    (check-equal? (tensor-shape ys) '(52 16))
    ;; xs is the text's first 52*16 ids in order; ys the same shifted by one
    (define id-list (map inexact->exact (tensor->list ids)))
    (check-equal? (map inexact->exact (tensor->list xs))
                  (take id-list (* 52 16)))
    (check-equal? (map inexact->exact (tensor->list ys))
                  (take (cdr id-list) (* 52 16)))
    (check-exn #rx"too short"
               (lambda () (contiguous-blocks (encode vocab "ab") 16))))

  ;; Full-corpus download path: fetch (cached) + strip, skipping when
  ;; Gutenberg is unreachable. When it runs, the stripped prose must start at
  ;; the title page (header gone), keep the novella text, and end before the
  ;; license (end matter gone).
  (define ok?
    (with-handlers ([exn:fail? (lambda (_) #f)])
      (define prose (load-heart-of-darkness))
      (and (regexp-match? #rx"^Heart of Darkness" prose)
           (string-contains? prose "The Nellie, a cruising yawl")
           (not (string-contains? prose "PROJECT GUTENBERG"))
           (> (string-length prose) 100000))))
  (if ok?
      (displayln "[text-test] load-heart-of-darkness OK (prose stripped)")
      (displayln "[text-test] skipped download (offline / Gutenberg unreachable)")))
