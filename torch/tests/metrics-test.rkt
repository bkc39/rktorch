#lang racket/base

(module+ test
  (require rackunit
           (only-in "../audio/functional.rkt" edit-distance)
           (only-in "../audio/librispeech.rkt" load-librispeech-fixture)
           (only-in "../audio/metrics.rkt" cer wer))

  (test-case "edit-distance matches the classic Levenshtein cases (#83)"
    (check-equal? (edit-distance (string->list "kitten")
                                 (string->list "sitting"))
                  3)
    (check-equal? (edit-distance '(a b c) '(a b c)) 0)
    (check-equal? (edit-distance '() '()) 0)
    (check-equal? (edit-distance '() '(a)) 1)
    (check-equal? (edit-distance '(a b c) '()) 3)
    (check-equal? (edit-distance '("the" "cat" "sat")
                                 '("the" "bat" "sat" "on"))
                  2)
    (check-exn exn:fail:contract?
               (lambda () (edit-distance "kitten" '()))))

  (test-case "wer is the exact word-error rate (#83)"
    (check-equal? (wer "the cat sat" "the cat sat") 0)
    (check-equal? (wer "the cat sat" "the bat sat on") 2/3)
    (check-equal? (wer "a b c d" "") 1)
    (check-exn #rx"transcript-with-words"
               (lambda () (wer "" "hello")))
    (check-exn #rx"transcript-with-words"
               (lambda () (wer "   " "hello"))))

  (test-case "cer is the exact character-error rate (#83)"
    (check-equal? (cer "abc" "abc") 0)
    (check-equal? (cer "abc" "axc") 1/3)
    (check-equal? (cer "ab" "") 1)
    (check-exn exn:fail:contract? (lambda () (cer "" "x"))))

  (test-case "metrics over the speech fixture transcript (#83)"
    (define-values (_samples _rate transcript) (load-librispeech-fixture))
    (check-equal? (wer transcript transcript) 0)
    (check-equal? (cer transcript transcript) 0)
    (check-true (> (wer transcript "MISTER QUILTER") 0))))
