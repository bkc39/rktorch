#lang racket/base

(module+ test
  (require rackunit
           (only-in "../audio/librispeech.rkt"
                    librispeech-utterances load-librispeech-fixture
                    load-utterance parse-trans-line utterance-id
                    utterance-path utterance-transcript)
           (only-in "../main.rkt" tensor-shape))

  (test-case "trans.txt lines parse to id + transcript (#83)"
    (check-equal? (parse-trans-line "1272-128104-0000 MISTER QUILTER IS")
                  '("1272-128104-0000" . "MISTER QUILTER IS"))
    (check-equal? (parse-trans-line "  84-121123-0001  A  B  ")
                  '("84-121123-0001" . "A  B"))
    (check-false (parse-trans-line ""))
    (check-false (parse-trans-line "   "))
    (check-exn #rx"no transcript"
               (lambda () (parse-trans-line "1272-128104-0000"))))

  (test-case "the committed utterance fixture loads with its transcript (#83)"
    (define-values (samples rate transcript) (load-librispeech-fixture))
    (check-equal? rate 16000)
    (check-equal? (car (tensor-shape samples)) 1)
    (check-true (> (cadr (tensor-shape samples)) 16000)
                "at least a second of speech")
    (check-equal?
     transcript
     "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL"))

  (when (getenv "RKTORCH_LIBRISPEECH_LIVE")
    (test-case "dev-clean enumerates and loads (live)"
      (define us (librispeech-utterances "dev-clean"))
      (check-true (> (length us) 2000))
      (define u (car us))
      (check-regexp-match #rx"^[0-9]+-[0-9]+-[0-9]+$" (utterance-id u))
      (check-true (file-exists? (utterance-path u)))
      (check-true (positive? (string-length (utterance-transcript u))))
      (define-values (samples rate) (load-utterance u))
      (check-equal? rate 16000)
      (check-equal? (car (tensor-shape samples)) 1))))
