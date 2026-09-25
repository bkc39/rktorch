#lang racket/base

(require (only-in ffi/vector
                  f32vector-set! f64vector-set! make-f32vector make-f64vector)
         (only-in racket/contract/base
                  -> ->* ->i and/c flat-named-contract list/c listof or/c
                  real-in)
         (only-in racket/math exact-floor exact-round exact-truncate)
         (only-in "../foreign.rkt"
                  copy! div draw-seed flip generator? matmul mul narrow ne
                  reshape select stack sub tensor tensor-device tensor-dtype
                  tensor-shape tensor? to-dtype transpose where zeros)
         (only-in "../foreign/contracts.rkt" image-batch/c)
         (only-in "../private/contract.rkt" define/contract-out))

;; One draw from the torch generator seeds a Racket generator for the batch,
;; so a seeded loader replays its augmentation while the images stay on the
;; device; random-seed takes 31 bits of the 63 drawn.
(define (batch-rng generator)
  (define rng (make-pseudo-random-generator))
  (parameterize ([current-pseudo-random-generator rng])
    (random-seed (bitwise-and (draw-seed #:generator generator) #x7FFFFFFF)))
  rng)

(define/contract-out (random-horizontal-flip x ;; noqa
                                             #:p [p 0.5]
                                             #:generator [generator #f])
  (->* [image-batch/c] [#:p (real-in 0 1) #:generator (or/c generator? #f)]
       tensor?)
  (define n (car (tensor-shape x)))
  (cond
    [(zero? n) x]
    [else
     (define rng (batch-rng generator))
     (define flipped
       (for/list ([_ (in-range n)]) (if (< (random rng) p) 1 0)))
     (define mask (reshape (ne (tensor flipped #:device (tensor-device x)) 0)
                           n 1 1 1))
     (where mask (flip x 3) x)]))

(define/contract-out (random-crop x ;; noqa
                                  #:padding [padding 4]
                                  #:generator [generator #f])
  (->* [image-batch/c]
       [#:padding exact-nonnegative-integer? #:generator (or/c generator? #f)]
       tensor?)
  (define dims (tensor-shape x))
  (define n (car dims))
  (cond
    [(zero? n) x]
    [else (cropped x dims n padding generator)]))

(define (cropped x dims n padding generator)
  (define h (caddr dims))
  (define w (cadddr dims))
  (define padded (zeros n (cadr dims) (+ h (* 2 padding)) (+ w (* 2 padding))
                        #:device (tensor-device x)
                        #:dtype (tensor-dtype x)))
  (copy! (narrow (narrow padded 2 padding h) 3 padding w) x)
  (define rng (batch-rng generator))
  (define span (add1 (* 2 padding)))
  (stack (for/list ([i (in-range n)])
           (define top (random span rng))
           (define left (random span rng))
           (narrow (narrow (select padded 0 i) 1 top h) 2 left w))
         0))

(define image-dtypes '(uint8 float16 bfloat16 float32 float64))
(define float-dtypes '(float16 bfloat16 float32 float64))

(define (spatial-size x)
  (define dims (reverse (tensor-shape x)))
  (values (cadr dims) (car dims)))

(define (channels x)
  (caddr (reverse (tensor-shape x))))

(define image-or-batch/c
  (flat-named-contract
   'image-or-batch
   (lambda (x)
     (and (tensor? x)
          (memv (length (tensor-shape x)) '(3 4))
          (andmap positive? (tensor-shape x))
          #t))))

(define float-image/c
  (flat-named-contract
   'float-image-or-batch
   (and/c image-or-batch/c
          (lambda (x) (and (memq (tensor-dtype x) float-dtypes) #t)))))

(define size/c
  (flat-named-contract
   'image-size
   (or/c exact-positive-integer?
         (list/c exact-positive-integer? exact-positive-integer?))))

(define (size->hw size)
  (if (pair? size) (values (car size) (cadr size)) (values size size)))

(define (crop-size/c x)
  (define-values (h w) (spatial-size x))
  (flat-named-contract
   'crop-size-within-the-image
   (and/c size/c
          (lambda (size)
            (define-values (ch cw) (size->hw size))
            (and (<= ch h) (<= cw w))))))

(define (channel-values/c x [element/c real?])
  (flat-named-contract
   'one-value-per-channel
   (and/c (listof element/c) (lambda (vs) (= (length vs) (channels x))))))

;; torchvision: an int is the short side's new length and the long side is
;; truncated to keep the aspect ratio
(define (output-size size h w)
  (cond
    [(pair? size) (values (car size) (cadr size))]
    [(<= w h) (values (quotient (* size h) w) size)]
    [else (values size (quotient (* size w) h))]))

(define (triangle x)
  (define a (abs x))
  (if (< a 1.0) (- 1.0 a) 0.0))

;; upsample_bilinear2d with align_corners=False: each output samples the
;; two inputs around its centre, the source index clamped at zero
(define (bilinear-row row in scale)
  (define src (max 0.0 (- (* scale (+ row 0.5)) 0.5)))
  (define i0 (min (exact-floor src) (sub1 in)))
  (define i1 (min (add1 i0) (sub1 in)))
  (define l1 (min (max (- src i0) 0.0) 1.0))
  (if (= i0 i1)
      (list (cons i0 1.0))
      (list (cons i0 (- 1.0 l1)) (cons i1 l1))))

;; _upsample_bilinear2d_aa, Pillow's filter: a downscale widens the
;; triangle to the scale, and each row is renormalized
(define (antialias-row row in scale)
  (define support (max scale 1.0))
  (define invscale (if (>= scale 1.0) (/ 1.0 scale) 1.0))
  (define center (* scale (+ row 0.5)))
  (define lo (max (exact-truncate (+ (- center support) 0.5)) 0))
  (define hi (min (exact-truncate (+ center support 0.5)) in))
  (define ws
    (for/list ([k (in-range lo hi)])
      (cons k (triangle (* (+ (- k center) 0.5) invscale)))))
  (define total (for/sum ([kw (in-list ws)]) (cdr kw)))
  (for/list ([kw (in-list ws)]) (cons (car kw) (/ (cdr kw) total))))

;; the weights are written at the image's own width, so the matrix is built
;; where the image lives with no float64 copy, which MPS could not hold
(define (interpolation-matrix in out antialias? dtype device)
  (define-values (make-weights put!)
    (if (eq? dtype 'float64)
        (values make-f64vector f64vector-set!)
        (values make-f32vector f32vector-set!)))
  (define weights (make-weights (* out in) 0.0))
  (define scale (exact->inexact (/ in out)))
  (define row-of (if antialias? antialias-row bilinear-row))
  (for* ([row (in-range out)]
         [kw (in-list (row-of row in scale))])
    (put! weights (+ (* row in) (car kw)) (cdr kw)))
  (reshape (tensor weights #:device device) out in))

(define (resized x oh ow antialias?)
  (define-values (h w) (spatial-size x))
  (define dtype (tensor-dtype x))
  (define device (tensor-device x))
  (define rows
    (if (= oh h)
        x
        (matmul (interpolation-matrix h oh antialias? dtype device) x)))
  (if (= ow w)
      rows
      (matmul rows
              (transpose (interpolation-matrix w ow antialias? dtype device)
                         0 1))))

(define/contract-out (resize x size #:antialias? [antialias? #t]) ;; noqa
  (->* [float-image/c size/c] [#:antialias? boolean?] tensor?)
  (define-values (h w) (spatial-size x))
  (define-values (oh ow) (output-size size h w))
  (define dtype (tensor-dtype x))
  ;; the half dtypes interpolate in float32, as ATen's CPU kernels do
  (if (memq dtype '(float16 bfloat16))
      (to-dtype (resized (to-dtype x 'float32) oh ow antialias?) dtype)
      (resized x oh ow antialias?)))

(define/contract-out (center-crop x size) ;; noqa
  (->i ([x image-or-batch/c] [size (x) (crop-size/c x)]) [result tensor?])
  (define-values (h w) (spatial-size x))
  (define-values (ch cw) (size->hw size))
  (define rank (length (tensor-shape x)))
  ;; Python's round, half to even, as torchvision's center_crop uses it
  (narrow (narrow x (- rank 2) (round (/ (- h ch) 2)) ch)
          (- rank 1) (round (/ (- w cw) 2)) cw))

(define/contract-out (normalize x mean std) ;; noqa
  (->i ([x float-image/c]
        [mean (x) (channel-values/c x)]
        [std (x) (channel-values/c x (and/c real? positive?))])
       [result tensor?])
  (define (per-channel vs)
    (reshape (tensor (map exact->inexact vs)
                     #:dtype (tensor-dtype x)
                     #:device (tensor-device x))
             (length vs) 1 1))
  (div (sub x (per-channel mean)) (per-channel std)))

(define/contract-out imagenet-mean (listof real?) '(0.485 0.456 0.406)) ;; noqa
(define/contract-out imagenet-std (listof real?) '(0.229 0.224 0.225)) ;; noqa

(define (three-channels/c name base/c)
  (flat-named-contract
   name
   (and/c base/c (lambda (img) (= 3 (channels img))))))

(define rgb-float-image/c
  (three-channels/c 'three-channel-float-image-or-batch float-image/c))

(define/contract-out (imagenet-normalize x) ;; noqa
  (-> rgb-float-image/c tensor?)
  (normalize x imagenet-mean imagenet-std))

(define image-dtype/c
  (flat-named-contract 'image-dtype
                       (lambda (d) (and (memq d image-dtypes) #t))))

(define image-tensor/c
  (flat-named-contract
   'image-dtype-tensor
   (lambda (t) (and (tensor? t) (memq (tensor-dtype t) image-dtypes) #t))))

(define/contract-out (convert-image-dtype x [dtype 'float32]) ;; noqa
  (->* [image-tensor/c] [image-dtype/c] tensor?)
  (define from (tensor-dtype x))
  (cond
    [(eq? from dtype) x]
    [(eq? from 'uint8) (div (to-dtype x dtype) 255)]
    ;; convert_image_dtype's scale: 1.0 lands on 255 and nothing past it
    [(eq? dtype 'uint8) (to-dtype (mul x (- 256.0 1e-3)) 'uint8)]
    [else (to-dtype x dtype)]))

(define rgb-image/c
  (three-channels/c 'three-channel-image-or-batch
                    (and/c image-or-batch/c image-tensor/c)))

(define/contract-out (imagenet-preprocess x) ;; noqa
  (-> rgb-image/c tensor?)
  (imagenet-normalize
   (center-crop (resize (convert-image-dtype x 'float32) 256) 224)))

(define (ordered-pair/c element/c)
  (flat-named-contract
   'ascending-pair
   (and/c (list/c element/c element/c)
          (lambda (pair) (<= (car pair) (cadr pair))))))

(define (crop-window rng h w scale ratio)
  (define (uniform lo hi) (+ lo (* (- hi lo) (random rng))))
  (define log-lo (log (exact->inexact (car ratio))))
  (define log-hi (log (exact->inexact (cadr ratio))))
  (or (for/or ([_ (in-range 10)])
        (define area (* h w (uniform (car scale) (cadr scale))))
        (define aspect (exp (uniform log-lo log-hi)))
        (define cw (exact-round (sqrt (* area aspect))))
        (define ch (exact-round (sqrt (/ area aspect))))
        (and (< 0 cw) (<= cw w) (< 0 ch) (<= ch h)
             (list (random (add1 (- h ch)) rng)
                   (random (add1 (- w cw)) rng)
                   ch cw)))
      (central-window h w ratio)))

(define (central-window h w ratio)
  (define-values (ch cw)
    (cond
      [(< (/ w h) (car ratio)) (values (exact-round (/ w (car ratio))) w)]
      [(> (/ w h) (cadr ratio)) (values h (exact-round (* h (cadr ratio))))]
      [else (values h w)]))
  (list (quotient (- h ch) 2) (quotient (- w cw) 2) ch cw))

(define (resized-crop image window size)
  (define rank (length (tensor-shape image)))
  (resize (narrow (narrow image (- rank 2) (car window) (caddr window))
                  (- rank 1) (cadr window) (cadddr window))
          (list size size)))

(define/contract-out (random-resized-crop x size ;; noqa
                                          #:scale [scale '(0.08 1.0)]
                                          #:ratio [ratio '(3/4 4/3)]
                                          #:generator [generator #f])
  (->* [float-image/c exact-positive-integer?]
       [#:scale (ordered-pair/c (and/c (real-in 0 1) positive?))
        #:ratio (ordered-pair/c (and/c real? positive?))
        #:generator (or/c generator? #f)]
       tensor?)
  (define rng (batch-rng generator))
  (define-values (h w) (spatial-size x))
  (define (one image)
    (resized-crop image (crop-window rng h w scale ratio) size))
  (if (= 3 (length (tensor-shape x)))
      (one x)
      (stack (for/list ([i (in-range (car (tensor-shape x)))])
               (one (select x 0 i)))
             0)))
