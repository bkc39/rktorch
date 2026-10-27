#lang racket/base

(module+ test
  (require rackunit
           (only-in "../audio/functional.rkt" log-mel-spectrogram mel-filterbank)
           (only-in "../audio/librispeech.rkt" load-librispeech-fixture)
           (only-in "../main.rkt" ref tensor->list to-dtype)
           (only-in "private/heap-allocation.rkt" median-allocation))

  (define blamed #rx"blaming: [(][^)]*log-mel-filterbank-test[.]rkt")

  (define (filterbank-violation? e)
    (and (exn:fail:contract? e)
         (regexp-match? blamed (exn-message e))
         (regexp-match? #rx"filterbank" (exn-message e))))

  (define-values (samples rate _transcript) (load-librispeech-fixture))
  (define wave (ref samples 0))
  (define bank (mel-filterbank #:n-freqs 201 #:n-mels 80 #:sample-rate rate))

  (test-case "L15: a given bank is the one the call would build"
    (check-equal?
     (tensor->list (log-mel-spectrogram wave #:sample-rate rate #:filterbank bank))
     (tensor->list (log-mel-spectrogram wave #:sample-rate rate))))

  (test-case "L15: a bank that does not fit is the caller's violation"
    (check-exn filterbank-violation?
               (lambda ()
                 (log-mel-spectrogram wave #:sample-rate rate #:n-mels 40
                                      #:filterbank bank)))
    (check-exn filterbank-violation?
               (lambda ()
                 (log-mel-spectrogram wave #:sample-rate rate #:n-fft 512
                                      #:filterbank bank)))
    (check-exn filterbank-violation?
               (lambda ()
                 (log-mel-spectrogram wave #:sample-rate rate
                                      #:filterbank (to-dtype bank 'float64)))))

  (test-case "L15: a call with the bank keeps the filters off the Racket heap"
    (define with-bank
      (median-allocation
       (lambda () (log-mel-spectrogram wave #:sample-rate rate #:filterbank bank))))
    (define without-bank
      (median-allocation (lambda () (log-mel-spectrogram wave #:sample-rate rate))))
    (check-true (< with-bank 262144) (format "~a bytes per call" with-bank))
    (check-true (<= (* 10 with-bank) without-bank)
                (format "~a bytes with the bank, ~a without" with-bank without-bank))))
