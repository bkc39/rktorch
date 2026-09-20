#lang racket/base

(module+ test
  (require rackunit
           (only-in "../audio/functional.rkt" hann-window mel-filterbank)
           (only-in "../main.rkt"
                    accelerator-if-available cpu-device dtype tensor-device
                    tensor-shape))

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

  ;; The auxiliaries are what #102 moves, so they are what this checks. The
  ;; whole pipeline cannot run here: stft on CUDA fails in cuFFT (#180), and
  ;; it fails the same way whether the window is moved or built in place.
  (unless (equal? accelerator (cpu-device))
    (test-case "the auxiliaries are built on the accelerator"
      (check-equal? (tensor-device
                     (hann-window 400 #:device accelerator #:dtype 'float32))
                    accelerator)
      (check-equal? (tensor-device
                     (mel-filterbank #:n-freqs 201 #:n-mels 80
                                     #:sample-rate rate #:device accelerator))
                    accelerator))))
