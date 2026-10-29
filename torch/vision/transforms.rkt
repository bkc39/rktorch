#lang racket/base

(require (only-in ffi/vector
                  f32vector-set! f64vector-set! make-f32vector make-f64vector
                  make-s64vector s64vector-set!)
         (only-in racket/contract/base
                  -> ->* ->i and/c flat-named-contract list/c listof or/c
                  real-in)
         (only-in racket/match match-define)
         (only-in racket/math exact-floor exact-round exact-truncate)
         (only-in threading ~>)
         (only-in "../foreign.rkt"
                  @ add arange clamp cpu-generator/c device device-type div
                  draw-seed dtype flip masked-fill! mul narrow ne reshape select
                  stack sub tensor tensor-shape tensor? to-dtype transpose where)
         (only-in "../foreign/contracts.rkt" image-batch/c)
         (only-in "../generated.rkt" index-tensor)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "private/contracts.rkt" ordered-pair/c))

(module+ private
  (provide batch-rng))

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
  (->* [image-batch/c]
       [#:p (real-in 0 1) #:generator (or/c cpu-generator/c #f)]
       tensor?)
  (define n (car (tensor-shape x)))
  (cond
    [(zero? n) x]
    [else
     (define rng (batch-rng generator))
     (define flipped
       (for/list ([_ (in-range n)]) (if (< (random rng) p) 1 0)))
     (define mask (reshape (ne (tensor flipped #:device (device x)) 0)
                           n 1 1 1))
     (where mask (flip x 3) x)]))

(define croppable-batch/c
  (flat-named-contract
   'image-batch-with-pixels
   (and/c image-batch/c
          (lambda (x)
            (match-define (list _ _ h w) (tensor-shape x))
            (and (positive? h) (positive? w))))))

(define/contract-out (random-crop x ;; noqa
                                  #:padding [padding 4]
                                  #:generator [generator #f])
  (->* [croppable-batch/c]
       [#:padding exact-nonnegative-integer?
        #:generator (or/c cpu-generator/c #f)]
       tensor?)
  (cond
    [(~> x tensor-shape car zero?) x]
    [else (cropped x padding generator)]))

(define (cropped x padding generator)
  (match-define (list n c h w) (tensor-shape x))
  (define dev (device x))
  (define rng (batch-rng generator))
  (define span (add1 (* 2 padding)))
  (define offsets (make-s64vector (* 2 n)))
  (for ([k (in-range (* 2 n))])
    (s64vector-set! offsets k (- (random span rng) padding)))
  (define corners (reshape (tensor offsets #:device dev) n 2))
  (define (positions extent)
    (arange extent #:device dev #:dtype 'int64))
  (define (starts axis)
    (~> corners (select 1 axis) (reshape n 1)))
  (define rows (add (starts 0) (reshape (positions h) 1 h)))
  (define cols (add (starts 1) (reshape (positions w) 1 w)))
  (define (inside raw extent)
    (~> raw (clamp #:min 0 #:max (sub1 extent)) (to-dtype 'int64)))
  (define rows* (inside rows h))
  (define cols* (inside cols w))
  (define batch (positions n))
  (define channel (positions c))
  ;; CUDA and MPS make every index contiguous at the broadcast shape when
  ;; their strides differ, so off the CPU each gather indexes two small dims
  (define out
    (cond
      [(eq? (device-type dev) 'cpu)
       (index-tensor x (list (reshape batch n 1 1 1)
                             (reshape channel 1 c 1 1)
                             (reshape rows* n 1 h 1)
                             (reshape cols* n 1 1 w)))]
      [else
       (~> x
           (index-tensor (list (reshape batch n 1) #f #f cols*))
           (index-tensor (list (reshape batch n 1 1)
                               #f
                               (reshape channel 1 c 1)
                               (reshape rows* n 1 h))))]))
  (masked-fill! out (reshape (ne rows rows*) n 1 h 1) 0)
  (masked-fill! out (reshape (ne cols cols*) n 1 1 w) 0)
  out)

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
          (lambda (x) (and (memq (dtype x) float-dtypes) #t)))))

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
(define (interpolation-matrix in out antialias? image)
  (define-values (make-weights put!)
    (if (eq? (dtype image) 'float64)
        (values make-f64vector f64vector-set!)
        (values make-f32vector f32vector-set!)))
  (define weights (make-weights (* out in) 0.0))
  (define scale (exact->inexact (/ in out)))
  (define row-of (if antialias? antialias-row bilinear-row))
  (for* ([row (in-range out)]
         [kw (in-list (row-of row in scale))])
    (put! weights (+ (* row in) (car kw)) (cdr kw)))
  (reshape (tensor weights #:device (device image)) out in))

(define (resized x oh ow antialias?)
  (define-values (h w) (spatial-size x))
  (define rows
    (if (= oh h)
        x
        (@ (interpolation-matrix h oh antialias? x) x)))
  (if (= ow w)
      rows
      (@ rows (transpose (interpolation-matrix w ow antialias? x) 0 1))))

(define/contract-out (resize x size #:antialias? [antialias? #t]) ;; noqa
  (->* [float-image/c size/c] [#:antialias? boolean?] tensor?)
  (define-values (h w) (spatial-size x))
  (define-values (oh ow) (output-size size h w))
  (define source-dtype (dtype x))
  ;; the half dtypes interpolate in float32, as ATen's CPU kernels do
  (if (memq source-dtype '(float16 bfloat16))
      (to-dtype (resized (to-dtype x 'float32) oh ow antialias?) source-dtype)
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
                     #:dtype (dtype x)
                     #:device (device x))
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
   (lambda (t) (and (tensor? t) (memq (dtype t) image-dtypes) #t))))

(define/contract-out (convert-image-dtype x [target-dtype 'float32]) ;; noqa
  (->* [image-tensor/c] [image-dtype/c] tensor?)
  (define from (dtype x))
  (cond
    [(eq? from target-dtype) x]
    [(eq? from 'uint8) (div (to-dtype x target-dtype) 255)]
    ;; convert_image_dtype's scale: 1.0 lands on 255 and nothing past it
    [(eq? target-dtype 'uint8) (to-dtype (mul x (- 256.0 1e-3)) 'uint8)]
    [else (to-dtype x target-dtype)]))

(define rgb-image/c
  (three-channels/c 'three-channel-image-or-batch
                    (and/c image-or-batch/c image-tensor/c)))

(define/contract-out (imagenet-preprocess x) ;; noqa
  (-> rgb-image/c tensor?)
  (imagenet-normalize
   (center-crop (resize (convert-image-dtype x 'float32) 256) 224)))

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
  (match-define (list lower upper) ratio)
  (define-values (ch cw)
    (cond
      [(< (/ w h) lower)
       (values (max 1 (exact-round (/ w lower))) w)]
      [(> (/ w h) upper)
       (values h (max 1 (exact-round (* h upper))))]
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
       [#:scale (~> (real-in 0 1) (and/c positive?) ordered-pair/c)
        #:ratio (ordered-pair/c (and/c real? positive?))
        #:generator (or/c cpu-generator/c #f)]
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
