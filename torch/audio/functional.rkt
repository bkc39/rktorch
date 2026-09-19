#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i =/c >=/c and/c any/c listof or/c
                  unsupplied-arg?)
         (only-in "../foreign/error.rkt" check-handle)
         (only-in "../foreign/ops.rkt" float-dtype/c placement)
         (only-in "../foreign/raw/spectral.rkt"
                  tr-hann-window/raw tr-stft/raw)
         (only-in "../foreign/structs.rkt" wrap-tensor)
         (only-in "../main.rkt"
                  add device/c dtype log matmul mul ref sqrt t tensor
                  tensor-device tensor? to-dtype)
         (only-in "../private/contract.rkt" define/contract-out))

(define maybe-length/c (or/c #f exact-positive-integer?))

(define/contract-out (hann-window window-length
                                  #:periodic? [periodic? #t]
                                  #:device [device #f]
                                  #:dtype [dtype #f])
  (->* (exact-nonnegative-integer?)
       (#:periodic? boolean?
        #:device (or/c #f device/c)
        #:dtype (or/c #f float-dtype/c))
       tensor?)
  (define-values (type index dt) (placement device dtype))
  (wrap-tensor
   (check-handle 'hann-window
                 (tr-hann-window/raw window-length periodic? type index dt))))

(define/contract-out (stft samples
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

(define/contract-out (spectrogram samples
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
        #:power (or/c (=/c 1) (=/c 2)))
       tensor?)
  (define frames (stft samples #:n-fft n-fft #:hop-length hop-length
                       #:win-length win-length #:window window
                       #:center? center?))
  (define re (ref frames .. 0))
  (define im (ref frames .. 1))
  (define magnitude-squared (add (mul re re) (mul im im)))
  (if (= power 2) magnitude-squared (sqrt magnitude-squared)))

;; twins torchaudio.functional.edit_distance
(define/contract-out (edit-distance reference hypothesis) ;; noqa
  (-> (listof any/c) (listof any/c) exact-nonnegative-integer?)
  (define hyp (list->vector hypothesis))
  (define n (vector-length hyp))
  (for/fold ([prev (build-vector (add1 n) values)]
             #:result (vector-ref prev n))
            ([r (in-list reference)]
             [i (in-naturals 1)])
    (define curr (make-vector (add1 n) i))
    (for ([h (in-vector hyp)]
          [j (in-naturals 1)])
      (vector-set! curr j
                   (min (add1 (vector-ref curr (sub1 j)))
                        (add1 (vector-ref prev j))
                        (+ (vector-ref prev (sub1 j))
                           (if (equal? r h) 0 1)))))
    curr))

(define (hz->mel f)
  (* 2595.0 (/ (log (+ 1.0 (/ f 700.0))) (log 10.0))))

(define (mel->hz m)
  (* 700.0 (- (expt 10.0 (/ m 2595.0)) 1.0)))

(define (linspace lo hi n)
  (for/list ([i (in-range n)])
    (+ lo (* (- hi lo) (/ i (exact->inexact (max 1 (sub1 n))))))))

;; HTK-scale triangular filters, torchaudio melscale_fbanks with
;; mel_scale "htk" and norm #f; result shape (n-freqs n-mels)
(define/contract-out (mel-filterbank #:n-freqs n-freqs
                                 #:n-mels n-mels
                                 #:sample-rate sample-rate
                                 #:f-min [f-min 0.0]
                                 #:f-max [f-max #f]
                                 #:device [device #f]
                                 #:dtype [dtype #f])
  (->i (#:n-freqs [n-freqs exact-positive-integer?]
        #:n-mels [n-mels exact-positive-integer?]
        #:sample-rate [sample-rate exact-positive-integer?])
       (#:f-min [f-min (and/c rational? (>=/c 0))]
        #:f-max [f-max (or/c #f (and/c rational? positive?))]
        #:device [device (or/c #f device/c)]
        #:dtype [dtype (or/c #f float-dtype/c)])
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
  (define rows
    (for/list ([f (in-list all-freqs)])
      (for/list ([m (in-range n-mels)])
        (define f-lo (vector-ref f-pts m))
        (define f-mid (vector-ref f-pts (add1 m)))
        (define f-hi (vector-ref f-pts (+ m 2)))
        (define down (/ (- f f-lo) (- f-mid f-lo)))
        (define up (/ (- f-hi f) (- f-hi f-mid)))
        (max 0.0 (min down up)))))
  ;; tensor builds no float64, so that one is a cast on the destination
  ;; rather than a second trip across the boundary
  (define buildable? (and (memq dtype '(#f float32)) #t))
  (define built (tensor rows #:device device #:dtype (and buildable? dtype)))
  (if buildable? built (to-dtype built dtype)))

(define/contract-out (log-mel-spectrogram samples ;; noqa
                                      #:sample-rate sample-rate
                                      #:n-fft [n-fft 400]
                                      #:hop-length [hop-length 160]
                                      #:n-mels [n-mels 80]
                                      #:eps [eps 1e-6])
  (->* (tensor? #:sample-rate exact-positive-integer?)
       (#:n-fft exact-positive-integer?
        #:hop-length exact-positive-integer?
        #:n-mels exact-positive-integer?
        #:eps (and/c rational? (>=/c 0)))
       tensor?)
  (define device (tensor-device samples))
  (define spec
    (spectrogram samples #:n-fft n-fft #:hop-length hop-length
                 #:window (hann-window n-fft
                                       #:device device
                                       #:dtype (dtype samples))))
  (define fb
    (mel-filterbank #:n-freqs (add1 (quotient n-fft 2))
                    #:n-mels n-mels
                    #:sample-rate sample-rate
                    #:device device
                    #:dtype (dtype spec)))
  (log (add (matmul (t fb 0 1) spec) eps)))
