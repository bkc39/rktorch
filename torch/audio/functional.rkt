#lang racket/base

(require racket/contract
         (only-in "../foreign/error.rkt" check-handle)
         (only-in "../foreign/raw/spectral.rkt"
                  tr-hann-window/raw tr-stft/raw)
         (only-in "../foreign/structs.rkt" wrap-tensor)
         (only-in "../main.rkt"
                  add dtype log matmul mul ref sqrt t tensor
                  tensor-device tensor? to-device to-dtype))

(provide hann-window
         log-mel-spectrogram
         mel-filterbank
         spectrogram
         stft)

(define maybe-length/c (or/c #f exact-positive-integer?))

(define/contract (hann-window window-length #:periodic? [periodic? #t])
  (->* (exact-nonnegative-integer?) (#:periodic? boolean?) tensor?)
  (wrap-tensor
   (check-handle 'hann-window
                 (tr-hann-window/raw window-length periodic?))))

(define/contract (stft samples
                       #:n-fft n-fft
                       #:hop-length [hop-length #f]
                       #:win-length [win-length #f]
                       #:window [window #f]
                       #:center? [center? #t]
                       #:normalized? [normalized? #f])
  (->* (tensor? #:n-fft exact-positive-integer?)
       (#:hop-length maybe-length/c
        #:win-length maybe-length/c
        #:window (or/c #f tensor?)
        #:center? boolean?
        #:normalized? boolean?)
       tensor?)
  (wrap-tensor
   (check-handle 'stft
                 (tr-stft/raw samples n-fft
                              (or hop-length -1) (or win-length -1)
                              window center? normalized?))))

(define/contract (spectrogram samples
                              #:n-fft n-fft
                              #:hop-length [hop-length #f]
                              #:win-length [win-length #f]
                              #:window [window #f]
                              #:center? [center? #t]
                              #:power [power 2.0])
  (->* (tensor? #:n-fft exact-positive-integer?)
       (#:hop-length maybe-length/c
        #:win-length maybe-length/c
        #:window (or/c #f tensor?)
        #:center? boolean?
        #:power (or/c 1.0 2.0))
       tensor?)
  (define frames (stft samples #:n-fft n-fft #:hop-length hop-length
                       #:win-length win-length #:window window
                       #:center? center?))
  (define re (ref frames .. 0))
  (define im (ref frames .. 1))
  (define magnitude-squared (add (mul re re) (mul im im)))
  (if (= power 2.0) magnitude-squared (sqrt magnitude-squared)))

(define (hz->mel f)
  (* 2595.0 (/ (log (+ 1.0 (/ f 700.0))) (log 10.0))))

(define (mel->hz m)
  (* 700.0 (- (expt 10.0 (/ m 2595.0)) 1.0)))

(define (linspace lo hi n)
  (for/list ([i (in-range n)])
    (+ lo (* (- hi lo) (/ i (exact->inexact (max 1 (sub1 n))))))))

;; HTK-scale triangular filters, torchaudio melscale_fbanks with
;; mel_scale "htk" and norm #f; result shape (n-freqs n-mels)
(define/contract (mel-filterbank #:n-freqs n-freqs
                                 #:n-mels n-mels
                                 #:sample-rate sample-rate
                                 #:f-min [f-min 0.0]
                                 #:f-max [f-max #f])
  (->i (#:n-freqs [n-freqs exact-positive-integer?]
        #:n-mels [n-mels exact-positive-integer?]
        #:sample-rate [sample-rate exact-positive-integer?])
       (#:f-min [f-min (and/c real? (>=/c 0))]
        #:f-max [f-max (or/c #f (and/c real? positive?))])
       #:pre/name (f-min f-max sample-rate) "f-min below the effective f-max"
       (< (if (unsupplied-arg? f-min) 0.0 f-min)
          (if (or (unsupplied-arg? f-max) (not f-max))
              (/ sample-rate 2.0)
              f-max))
       [result tensor?])
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

(define/contract (log-mel-spectrogram samples
                                      #:sample-rate sample-rate
                                      #:n-fft [n-fft 400]
                                      #:hop-length [hop-length 160]
                                      #:n-mels [n-mels 80]
                                      #:eps [eps 1e-6])
  (->* (tensor? #:sample-rate exact-positive-integer?)
       (#:n-fft exact-positive-integer?
        #:hop-length exact-positive-integer?
        #:n-mels exact-positive-integer?
        #:eps (and/c real? (>=/c 0)))
       tensor?)
  (define device (tensor-device samples))
  (define spec
    (spectrogram samples #:n-fft n-fft #:hop-length hop-length
                 #:window (to-dtype (to-device (hann-window n-fft) device)
                                    (dtype samples))))
  (define fb
    (mel-filterbank #:n-freqs (add1 (quotient n-fft 2))
                    #:n-mels n-mels
                    #:sample-rate sample-rate))
  (define fb-matched
    (to-dtype (to-device fb device) (dtype spec)))
  (log (add (matmul (t fb-matched 0 1) spec) eps)))
