#lang racket/base

(module+ test
  (require rackunit
           (only-in "../audio/functional.rkt"
                    hann-window log-mel-spectrogram mel-filterbank
                    spectrogram stft)
           (only-in "../audio/librispeech.rkt" load-librispeech-fixture)
           (only-in "../main.rkt"
                    item max min ones ref tensor-shape tensor->list))

  (test-case "hann-window matches the closed form (#83)"
    (check-equal? (tensor->list (hann-window 4)) '(0.0 0.5 1.0 0.5))
    (define symmetric (tensor->list (hann-window 4 #:periodic? #f)))
    (check-equal? (car symmetric) 0.0)
    (check-= (cadr (reverse symmetric)) (cadr symmetric) 1e-6)
    (check-equal? (tensor->list (hann-window 0)) '())
    (check-exn #rx"window length"
               (lambda () (hann-window -1)))
    (check-exn #rx"window length"
               (lambda () (hann-window 4.0))))

  (test-case "stft of a constant concentrates at DC (#83)"
    (define frames
      (stft (ones 8) #:n-fft 4 #:hop-length 4 #:center? #f))
    (check-equal? (tensor-shape frames) '(3 2 2))
    (define v (tensor->list frames))
    (check-equal? (car v) 4.0)
    (for ([x (in-list (list-tail v 4))])
      (check-= x 0.0 1e-6))
    (define windowed
      (stft (ones 4) #:n-fft 4 #:hop-length 4 #:win-length 4
            #:window (hann-window 4) #:center? #f))
    (check-equal? (car (tensor->list windowed)) 2.0)
    (check-exn #rx"n-fft" (lambda () (stft (ones 8) #:n-fft 0)))
    (check-exn #rx"window must be"
               (lambda () (stft (ones 8) #:n-fft 4 #:window '(1 2)))))

  (test-case "spectrogram power modes (#83)"
    (define p2 (spectrogram (ones 8) #:n-fft 4 #:hop-length 4 #:center? #f))
    (check-equal? (tensor-shape p2) '(3 2))
    (check-equal? (car (tensor->list p2)) 16.0)
    (define p1 (spectrogram (ones 8) #:n-fft 4 #:hop-length 4 #:center? #f
                            #:power 1.0))
    (check-equal? (car (tensor->list p1)) 4.0)
    (check-exn #rx"power must be"
               (lambda () (spectrogram (ones 8) #:n-fft 4 #:power 3.0))))

  (test-case "mel filterbank triangles are bounded and ordered (#83)"
    (define fb (mel-filterbank #:n-freqs 201 #:n-mels 80
                               #:sample-rate 16000))
    (check-equal? (tensor-shape fb) '(201 80))
    (check-true (>= (item (min fb)) 0.0))
    (check-true (<= (item (max fb)) 1.0))
    (check-true (positive? (item (max fb)))))

  (test-case "log-mel over the speech fixture (#83)"
    (define-values (samples rate _transcript) (load-librispeech-fixture))
    (define n (cadr (tensor-shape samples)))
    (define log-mel
      (log-mel-spectrogram (ref samples 0) #:sample-rate rate))
    (check-equal? (tensor-shape log-mel)
                  (list 80 (add1 (quotient n 160))))))
