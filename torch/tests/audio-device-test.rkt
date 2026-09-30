#lang racket/base

(module+ test
  (require rackunit
           (only-in "../audio/functional.rkt"
                    hann-window log-mel-spectrogram mel-filterbank stft)
           (only-in "../main.rkt"
                    accelerator-if-available cpu-device dtype manual-seed! randn
                    tensor->list tensor-device tensor-shape to-device))

  (define rate 16000)

  (test-case "hann-window is built where it is asked for"
    (define w (hann-window 8 #:device (cpu-device) #:dtype 'float64))
    (check-equal? (tensor-device w) (cpu-device))
    (check-equal? (dtype w) 'float64)
    (check-equal? (tensor-shape w) '(8)))

  (test-case "the filterbank is built where it is asked for"
    (define fb (mel-filterbank #:n-freqs 201 #:n-mels 80 #:sample-rate rate
                               #:device (cpu-device)))
    (check-equal? (tensor-device fb) (cpu-device))
    (check-equal? (dtype fb) 'float32)
    ;; float64 is past what `tensor` builds, so this exercises the cast
    (define wide (mel-filterbank #:n-freqs 16 #:n-mels 4 #:sample-rate rate
                                 #:device (cpu-device) #:dtype 'float64))
    (check-equal? (dtype wide) 'float64)
    (check-equal? (tensor-shape wide) '(16 4)))

  (define accelerator (accelerator-if-available))

  (define (agrees-with-cpu on-device on-cpu tol)
    (check-equal? (tensor-shape on-device) (tensor-shape on-cpu))
    (for ([a (in-list (tensor->list (to-device on-device (cpu-device))))]
          [c (in-list (tensor->list on-cpu))])
      (check-= a c tol)))

  (unless (equal? accelerator (cpu-device))
    (test-case "the auxiliaries are built on the accelerator"
      (check-equal? (tensor-device
                     (hann-window 400 #:device accelerator #:dtype 'float32))
                    accelerator)
      (check-equal? (tensor-device
                     (mel-filterbank #:n-freqs 201 #:n-mels 80
                                     #:sample-rate rate #:device accelerator))
                    accelerator))

    (test-case "stft and the log-mel front end run on the accelerator (#180)"
      (manual-seed! 0)
      (define samples (randn rate))
      (define (frames dev)
        (stft (to-device samples dev) #:n-fft 400 #:hop-length 160
              #:window (hann-window 400 #:device dev)))
      (define on-accelerator (frames accelerator))
      (check-equal? (tensor-device on-accelerator) accelerator)
      (agrees-with-cpu on-accelerator (frames (cpu-device)) 1e-3)
      (define (mels dev)
        (log-mel-spectrogram (to-device samples dev) #:sample-rate rate))
      (define mels-on-accelerator (mels accelerator))
      (check-equal? (tensor-device mels-on-accelerator) accelerator)
      (agrees-with-cpu mels-on-accelerator (mels (cpu-device)) 1e-3))))
