#lang racket/base

(require (only-in "../foreign/error.rkt" check-handle)
         (only-in "../foreign/raw/spectral.rkt"
                  tr-hann-window/raw tr-stft/raw)
         (only-in "../foreign/structs.rkt" wrap-tensor)
         (only-in "../main.rkt"
                  add log matmul mul ref sqrt t tensor tensor?))

(provide hann-window
         log-mel-spectrogram
         mel-filterbank
         spectrogram
         stft)

(define (hann-window window-length #:periodic? [periodic? #t])
  (unless (exact-nonnegative-integer? window-length)
    (error 'hann-window
           "window length must be an exact nonnegative integer: ~e"
           window-length))
  (wrap-tensor
   (check-handle 'hann-window
                 (tr-hann-window/raw window-length periodic?))))

(define (stft samples
              #:n-fft n-fft
              #:hop-length [hop-length #f]
              #:win-length [win-length #f]
              #:window [window #f]
              #:center? [center? #t]
              #:normalized? [normalized? #f])
  (unless (exact-positive-integer? n-fft)
    (error 'stft "n-fft must be an exact positive integer: ~e" n-fft))
  (unless (or (not hop-length) (exact-positive-integer? hop-length))
    (error 'stft "hop-length must be #f or an exact positive integer: ~e"
           hop-length))
  (unless (or (not win-length) (exact-positive-integer? win-length))
    (error 'stft "win-length must be #f or an exact positive integer: ~e"
           win-length))
  (unless (or (not window) (tensor? window))
    (error 'stft "window must be #f or a tensor: ~e" window))
  (wrap-tensor
   (check-handle 'stft
                 (tr-stft/raw samples n-fft
                              (or hop-length -1) (or win-length -1)
                              window center? normalized?))))

(define (spectrogram samples
                     #:n-fft n-fft
                     #:hop-length [hop-length #f]
                     #:win-length [win-length #f]
                     #:window [window #f]
                     #:center? [center? #t]
                     #:power [power 2.0])
  (define frames (stft samples #:n-fft n-fft #:hop-length hop-length
                       #:win-length win-length #:window window
                       #:center? center?))
  (define re (ref frames .. 0))
  (define im (ref frames .. 1))
  (define magnitude-squared (add (mul re re) (mul im im)))
  (cond
    [(= power 2.0) magnitude-squared]
    [(= power 1.0) (sqrt magnitude-squared)]
    [else (error 'spectrogram "power must be 1.0 or 2.0: ~e" power)]))

(define (hz->mel f)
  (* 2595.0 (/ (log (+ 1.0 (/ f 700.0))) (log 10.0))))

(define (mel->hz m)
  (* 700.0 (- (expt 10.0 (/ m 2595.0)) 1.0)))

(define (linspace lo hi n)
  (for/list ([i (in-range n)])
    (+ lo (* (- hi lo) (/ i (exact->inexact (max 1 (sub1 n))))))))

;; HTK-scale triangular filters, torchaudio melscale_fbanks with
;; mel_scale "htk" and norm #f; result shape (n-freqs n-mels)
(define (mel-filterbank #:n-freqs n-freqs
                        #:n-mels n-mels
                        #:sample-rate sample-rate
                        #:f-min [f-min 0.0]
                        #:f-max [f-max #f])
  (unless (exact-positive-integer? n-freqs)
    (error 'mel-filterbank
           "n-freqs must be an exact positive integer: ~e" n-freqs))
  (unless (exact-positive-integer? n-mels)
    (error 'mel-filterbank
           "n-mels must be an exact positive integer: ~e" n-mels))
  (unless (exact-positive-integer? sample-rate)
    (error 'mel-filterbank
           "sample rate must be an exact positive integer: ~e" sample-rate))
  (define hi (or f-max (/ sample-rate 2.0)))
  (define all-freqs
    (linspace 0.0 (exact->inexact (quotient sample-rate 2)) n-freqs))
  (define m-pts (linspace (hz->mel f-min) (hz->mel hi) (+ n-mels 2)))
  (define f-pts (for/vector ([m (in-list m-pts)]) (mel->hz m)))
  (tensor
   (for/list ([f (in-list all-freqs)])
     (for/list ([m (in-range n-mels)])
       (define f-lo (vector-ref f-pts m))
       (define f-mid (vector-ref f-pts (add1 m)))
       (define f-hi (vector-ref f-pts (+ m 2)))
       (define down (/ (- f f-lo) (- f-mid f-lo)))
       (define up (/ (- f-hi f) (- f-hi f-mid)))
       (max 0.0 (min down up))))))

(define (log-mel-spectrogram samples
                             #:sample-rate sample-rate
                             #:n-fft [n-fft 400]
                             #:hop-length [hop-length 160]
                             #:n-mels [n-mels 80]
                             #:eps [eps 1e-6])
  (define spec
    (spectrogram samples #:n-fft n-fft #:hop-length hop-length
                 #:window (hann-window n-fft)))
  (define fb
    (mel-filterbank #:n-freqs (add1 (quotient n-fft 2))
                    #:n-mels n-mels
                    #:sample-rate sample-rate))
  (log (add (matmul (t fb 0 1) spec) eps)))
