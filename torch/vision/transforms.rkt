#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i and/c flat-contract-predicate flat-named-contract
                  list/c listof or/c real-in)
         (only-in racket/flonum flsingle)
         (only-in racket/match match match-define)
         (only-in racket/math exact-round)
         (only-in threading ~>)
         (only-in "../foreign.rkt" - cpu-generator/c)
         (only-in (submod "../foreign.rkt" unchecked)
                  add arange clamp cpu-capability device device-type draw-seed
                  dtype flip masked-fill! mul narrow ne permute rand reshape
                  select stack sub! tensor tensor->list tensor-shape tensor?
                  to to-dtype where zeros)
         (only-in "../generated.rkt"
                  constant-pad-nd div-scalar! div-tensor! index-tensor
                  random-from! round-tensor upsample-bilinear2d
                  upsample-bilinear2d-aa)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "private/contracts.rkt"
                  default-ratio default-scale ratio/c scale/c))

(module+ private
  (provide batch-rng native-uint8-resize? resize/path resized-crop))

;; One draw from the torch generator seeds a Racket generator for the batch,
;; so a seeded loader replays its augmentation while the images stay on the
;; device; random-seed takes 31 bits of the 63 drawn.
(define (batch-rng generator)
  (define rng (make-pseudo-random-generator))
  (parameterize ([current-pseudo-random-generator rng])
    (~> (draw-seed #:generator generator) (bitwise-and #x7FFFFFFF) random-seed))
  rng)

(define (coins n p generator)
  (define threshold (flsingle (exact->inexact p)))
  (for/list ([u (in-list (tensor->list (rand n #:device 'cpu
                                             #:generator generator)))])
    (< u threshold)))

(define image-or-image-batch/c
  (flat-named-contract
   'image-or-image-batch
   (lambda (x)
     (and (tensor? x) (~> (tensor-shape x) length (memv '(3 4))) #t))))

(define image-or-batch/c
  (flat-named-contract
   'image-or-batch
   (and/c image-or-image-batch/c
          (lambda (x) (andmap positive? (tensor-shape x))))))

(define (image-count x)
  (match (tensor-shape x)
    [(list _ _ _) 1]
    [(cons n _) n]))

(define (last-axis x)
  (~> (tensor-shape x) length sub1))

(define/contract-out (random-horizontal-flip x ;; noqa
                                             #:p [p 0.5]
                                             #:generator [generator #f])
  (->* [image-or-image-batch/c]
       [#:p (real-in 0 1) #:generator (or/c cpu-generator/c #f)]
       tensor?)
  (define n (image-count x))
  (define heads (coins n p generator))
  (cond
    [(not (memq #t heads)) x]
    [(andmap values heads) (flip x (last-axis x))]
    [else
     (define mask
       (~> (for/list ([head? (in-list heads)]) (if head? 1 0))
           (tensor #:device (device x))
           (ne 0)
           (reshape n 1 1 1)))
     (where mask (flip x 3) x)]))

(define croppable/c
  (flat-named-contract
   'image-or-image-batch-with-pixels
   (and/c image-or-image-batch/c
          (lambda (x)
            (match-define (list _ ... h w) (tensor-shape x))
            (and (positive? h) (positive? w))))))

(define/contract-out (random-crop x ;; noqa
                                  #:padding [padding 4]
                                  #:generator [generator #f])
  (->* [croppable/c]
       [#:padding exact-nonnegative-integer?
        #:generator (or/c cpu-generator/c #f)]
       tensor?)
  (define n (image-count x))
  (cond
    [(or (zero? n) (zero? padding)) x]
    [else
     (define corners
       (~> (zeros n 2 #:device 'cpu #:dtype 'int64)
           (random-from! 0 (add1 (* 2 padding)) generator)))
     (if (= n 1)
         (windowed x padding corners)
         (gathered x padding corners))]))

(define (windowed x padding corners)
  (match-define (list _ ... h w) (tensor-shape x))
  (match-define (list top left) (tensor->list corners))
  (define width-axis (last-axis x))
  (~> x
      (constant-pad-nd (list padding padding padding padding) 0.0)
      (narrow (sub1 width-axis) top h)
      (narrow width-axis left w)))

(define (gathered x padding corners)
  (match-define (list n c h w) (tensor-shape x))
  (define dev (device x))
  (define starts (~> corners (- padding) (to dev)))
  (define (positions extent)
    (arange extent #:device dev #:dtype 'int64))
  (define (along axis extent)
    (add (~> starts (select 1 axis) (reshape n 1))
         (reshape (positions extent) 1 extent)))
  (define rows (along 0 h))
  (define cols (along 1 w))
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

(define float-image/c
  (flat-named-contract
   'float-image-or-batch
   (and/c image-or-batch/c
          (lambda (x) (and (memq (dtype x) float-dtypes) #t)))))

(define image-dtype/c
  (flat-named-contract 'image-dtype
                       (lambda (d) (and (memq d image-dtypes) #t))))

(define image-tensor/c
  (flat-named-contract
   'image-dtype-tensor
   (lambda (t) (and (tensor? t) (memq (dtype t) image-dtypes) #t))))

(define resizable-image/c
  (flat-named-contract
   'uint8-or-float-image-or-batch
   (and/c image-or-batch/c image-tensor/c)))

(define size/c
  (flat-named-contract
   'image-size
   (or/c exact-positive-integer?
         (list/c exact-positive-integer? exact-positive-integer?))))

(define (size->hw size)
  (if (pair? size) (values (car size) (cadr size)) (values size size)))

(define image-size? (flat-contract-predicate size/c))

(define (fits? size h w)
  (define-values (ch cw) (size->hw size))
  (and (<= ch h) (<= cw w)))

(define (crop-size/c x)
  (define-values (h w) (spatial-size x))
  (flat-named-contract
   'crop-size-within-image
   (lambda (size) (and (image-size? size) (fits? size h w)))))

;; torchvision: an int is the short side's new length and the long side is
;; truncated to keep the aspect ratio
(define (output-size size h w)
  (cond
    [(pair? size) (values (car size) (cadr size))]
    [(<= w h) (values (quotient (* size h) w) size)]
    [else (values size (quotient (* size w) h))]))

(define native-uint8-resize?
  (and (or (memq (cpu-capability) '(avx2 avx512))
           (eq? (system-type 'arch) 'aarch64))
       #t))

(define (uint8-path x)
  (if (and native-uint8-resize? (~> x device device-type (eq? 'cpu)))
      'native
      'float))

(define (interpolated x oh ow antialias?)
  ((if antialias? upsample-bilinear2d-aa upsample-bilinear2d)
   x (list oh ow) #f #f #f))

(define (channels-last x n c h w)
  (~> x (permute 0 2 3 1) (reshape -1) (reshape n h w c) (permute 0 3 1 2)))

(define (batched x oh ow antialias? [native-uint8? #f])
  (match (tensor-shape x)
    [(list c h w)
     (~> x (reshape 1 c h w) (batched oh ow antialias? native-uint8?) (reshape c oh ow))]
    [(list n 3 h w)
     #:when (and native-uint8? (= oh 1) (= ow 1))
     (~> x (channels-last n 3 h w) (interpolated oh ow antialias?))]
    [_ (interpolated x oh ow antialias?)]))

(define (through-float32 x source oh ow antialias? narrow-back)
  (~> x
      (to-dtype 'float32)
      (batched oh ow antialias?)
      narrow-back
      (to-dtype source)))

(define (resampled x oh ow antialias? path)
  (define source (dtype x))
  (cond
    [(memq source '(float16 bfloat16))
     (through-float32 x source oh ow antialias? values)]
    [(and (eq? source 'uint8) (eq? (or path (uint8-path x)) 'float))
     (through-float32 x source oh ow antialias? round-tensor)]
    [else (batched x oh ow antialias? (eq? source 'uint8))]))

(define (resize/path x size antialias? [path #f])
  (define-values (h w) (spatial-size x))
  (define-values (oh ow) (output-size size h w))
  (if (and (= oh h) (= ow w))
      x
      (resampled x oh ow antialias? path)))

(define/contract-out (resize x size #:antialias? [antialias? #t]) ;; noqa
  (->* [resizable-image/c size/c] [#:antialias? boolean?] tensor?)
  (resize/path x size antialias?))

(define/contract-out (center-crop x size) ;; noqa
  (->i ([x image-or-batch/c] [size (x) (crop-size/c x)]) [result tensor?])
  (define-values (h w) (spatial-size x))
  (define-values (ch cw) (size->hw size))
  (define rank (length (tensor-shape x)))
  ;; Python's round, half to even, as torchvision's center_crop uses it
  (narrow (narrow x (- rank 2) (~> (- h ch) (/ 2) round) ch)
          (- rank 1) (~> (- w cw) (/ 2) round) cw))

(define (one-per-channel? vs x)
  (define n (length vs))
  (or (= n 1) (= n (channels x))))

(define (channel-tensor vs dt dev)
  (tensor (map (lambda (v) (list (list (exact->inexact v)))) vs)
          #:dtype dt
          #:device dev))

(define/contract-out (normalize x mean std #:inplace? [inplace? #f]) ;; noqa
  (->i ([x float-image/c] [mean (listof real?)] [std (listof real?)])
       (#:inplace? [inplace? boolean?])
       #:pre/name (x mean) "one mean value per channel, or one for all"
       (one-per-channel? mean x)
       #:pre/name (x std) "one std value per channel, or one for all"
       (one-per-channel? std x)
       #:pre/name (std) "std evaluated to zero, leading to division by zero"
       (not (memf zero? std))
       [result tensor?])
  (define dt (dtype x))
  (define dev (device x))
  (define m (channel-tensor mean dt dev))
  (define s (channel-tensor std dt dev))
  (cond
    [inplace? (sub! x m) (div-tensor! x s)]
    [else (div-tensor! (- x m) s)]))

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

(define/contract-out (convert-image-dtype x [target-dtype 'float32]) ;; noqa
  (->* [image-tensor/c] [image-dtype/c] tensor?)
  (define from (dtype x))
  (cond
    [(eq? from target-dtype) x]
    [(eq? from 'uint8) (div-scalar! (to-dtype x target-dtype) 255.0)]
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

(define (resized-crop image window size #:antialias? [antialias? #t])
  (match-define (list top left height width) window)
  (define rank (length (tensor-shape image)))
  (~> image
      (narrow (- rank 2) top height)
      (narrow (- rank 1) left width)
      (resize/path size antialias?)))

(define/contract-out (random-resized-crop x size ;; noqa
                                          #:scale [scale default-scale]
                                          #:ratio [ratio default-ratio]
                                          #:antialias? [antialias? #t]
                                          #:generator [generator #f])
  (->* [resizable-image/c exact-positive-integer?]
       [#:scale scale/c
        #:ratio ratio/c
        #:antialias? boolean?
        #:generator (or/c cpu-generator/c #f)]
       tensor?)
  (define rng (batch-rng generator))
  (define-values (h w) (spatial-size x))
  (define (one image)
    (resized-crop image (crop-window rng h w scale ratio) (list size size)
                  #:antialias? antialias?))
  (if (= 3 (length (tensor-shape x)))
      (one x)
      (stack (for/list ([i (in-range (car (tensor-shape x)))])
               (one (select x 0 i)))
             0)))
