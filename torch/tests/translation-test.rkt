#lang racket/base

(module+ test
  (require rackunit
           (only-in file/zip zip)
           (only-in racket/file
                    display-to-file file->bytes make-directory*
                    make-temporary-directory delete-directory/files)
           (only-in racket/list first last)
           "../data/translation.rkt"
           "../main.rkt")

  (test-case "normalize-sentence follows the tutorial's normalizeString"
    (check-equal? (normalize-sentence "Go.") "go")
    (check-equal? (normalize-sentence "  I'm OK!  ") "i m ok !")
    (check-equal? (normalize-sentence "Qu'est-il arrivé à cet ami ?")
                  "qu est il arrive a cet ami ?")
    (check-equal? (normalize-sentence "Ça coûte 20 €, n'est-ce pas?")
                  "ca coute n est ce pas ?")
    (check-equal? (strip-accents "préféré Noël garçon") "prefere Noel garcon"))

  (test-case "parse-pairs filters by length and English prefix"
    (define text
      (string-append
       "I am cold.\tJ'ai froid.\n"
       "Go.\tVa !\n"
       "He is a very very very very very very tall man.\tIl est grand.\n"
       "We're here.\tNous sommes ici.\tattribution column\n"
       "malformed line without a tab\n"))
    (check-equal? (parse-pairs text)
                  '(("j ai froid" . "i am cold")
                    ("nous sommes ici" . "we re here")))
    (check-equal? (parse-pairs text #:source 'eng)
                  '(("i am cold" . "j ai froid")
                    ("we re here" . "nous sommes ici")))
    (check-equal? (length (parse-pairs text #:prefixes #f)) 3)
    (check-equal? (length (parse-pairs text #:prefixes #f #:max-length 12)) 4))

  (test-case "the committed fixture parses to the oracle's counts"
    (define pairs (load-translation-fixture))
    (check-equal? (length pairs) 287)
    (check-equal? (first pairs) '("je vais bien" . "i m ok"))
    (check-equal? (last pairs)
                  '("j en ai plutot marre de conduire chaque matin"
                    . "i m getting pretty bored with driving every morning"))
    (define-values (fra eng) (pairs->vocabs pairs))
    (check-equal? (vocab-size fra) (+ 3 509))
    (check-equal? (vocab-size eng) (+ 3 443))
    (check-equal? (car (load-translation-fixture #:source 'eng))
                  '("i m ok" . "je vais bien")))

  (test-case "vocabularies index words in order of first appearance"
    (define v (words->vocab '("je vais bien" "je suis la")))
    (check-equal? (vector->list (word-vocab-words v))
                  '("<pad>" "<sos>" "<eos>" "je" "vais" "bien" "suis" "la"))
    (check-equal? (list pad-id sos-id eos-id) '(0 1 2))
    (check-equal? (encode-sentence v "je suis bien") '(3 6 5 2))
    (check-exn #rx"word not in the vocabulary"
               (lambda () (encode-sentence v "je suis perdu"))))

  (test-case "a vocabulary's words cannot be edited under its id map"
    (define v (words->vocab '("je vais bien")))
    (check-pred immutable? (word-vocab-words v))
    (check-exn exn:fail:contract?
               (lambda () (vector-set! (word-vocab-words v) 3 "tu"))))

  (test-case "only a complete archive holding the pairs is accepted"
    (define dir (make-temporary-directory))
    (define whole (build-path dir "whole.zip"))
    (define cut (build-path dir "cut.zip"))
    (parameterize ([current-directory dir])
      (make-directory* "data")
      (display-to-file "I am cold.\tJ'ai froid.\n" "data/eng-fra.txt")
      (display-to-file "other" "data/other.txt")
      (zip whole "data/eng-fra.txt")
      (zip (build-path dir "other.zip") "data/other.txt"))
    (define bytes (file->bytes whole))
    (call-with-output-file cut
      (lambda (out)
        (write-bytes bytes out 0 (quotient (bytes-length bytes) 2))))
    (check-true (translation-archive? whole))
    (check-false (translation-archive? cut))
    (check-false (translation-archive? (build-path dir "other.zip")))
    (check-false (translation-archive? (build-path dir "missing.zip")))
    (delete-directory/files dir))

  (test-case "decode-tokens stops at <eos> and skips the other specials"
    (define v (words->vocab '("je vais bien")))
    (check-equal? (decode-tokens v '(1 3 4 5 2 0 0)) "je vais bien")
    (check-equal? (decode-tokens v '(3 2 4)) "je")
    (check-equal? (decode-tokens v '(2)) "")
    (check-equal? (decode-tokens v (to-dtype (tensor '(3 5 2)) 'int64))
                  "je bien"))

  (test-case "sentences pad to a common width after their <eos>"
    (define v (words->vocab '("je vais bien" "oui")))
    (define t (sentences->tensor v '("je vais bien" "oui")))
    (check-equal? (tensor-dtype t) 'int64)
    (check-equal? (tensor-shape t) '(2 4))
    (check-equal? (tensor->list t) '(3 4 5 2 6 2 0 0))
    (check-equal? (tensor-shape (sentences->tensor v '("oui") #:width 10))
                  '(1 10))
    (check-exn #rx"longer than the width"
               (lambda () (sentences->tensor v '("je vais bien") #:width 3))))

  (test-case "pairs->tensors encodes each side with its own vocabulary"
    (define pairs (load-translation-fixture))
    (define-values (fra eng) (pairs->vocabs pairs))
    (define-values (xs ys) (pairs->tensors pairs fra eng #:width 10))
    (check-equal? (tensor-shape xs) '(287 10))
    (check-equal? (tensor-shape ys) '(287 10))
    (check-equal? (decode-tokens fra (narrow xs 0 0 1)) "je vais bien")
    (check-equal? (decode-tokens eng (narrow ys 0 0 1)) "i m ok"))

  ;; The whole corpus, when RKTORCH_TRANSLATION_FULL is set (it downloads 3 MB
  ;; once): the counts are the PyTorch tutorial's, minus its two specials.
  (when (getenv "RKTORCH_TRANSLATION_FULL")
    (test-case "the full corpus filters to the tutorial's 11445 pairs"
      (define pairs (load-translation-pairs))
      (check-equal? (length pairs) 11445)
      (define-values (fra eng) (pairs->vocabs pairs))
      (check-equal? (vocab-size fra) (+ 3 4599))
      (check-equal? (vocab-size eng) (+ 3 2989))
      (check-equal? (last pairs)
                    '("c est un representant accredite du gouvernement canadien"
                      . "he s an accredited representative of the canadian government")))))
