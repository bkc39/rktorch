#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i >=/c and/c any any/c contract non-empty-listof
                  listof or/c real-in)
         (only-in racket/match match-define)
         (only-in threading ~>)
         (only-in "../foreign.rkt"
                  * + - / < affine-grid cat clamp device dtype exp grid-sample
                  log rand reshape select shape sqrt stack tensor tensor?
                  to-dtype where zeros-like)
         (only-in "../foreign/generator.rkt" generator-on?)
         (only-in "../generated.rkt" broadcast-to)
         (only-in "../private/contract.rkt" define/contract-out supplied-or)
         (only-in "private/color.rkt"
                  blend brightened column contrast-mean hue-turned luma
                  solarized)
         (only-in "private/contracts.rkt"
                  crop-boxes/c float-batch/c generator-for/c image-size/c
                  jitter-strengths/c ordered-pair/c per-sample-mask/c
                  per-sample/c rgb-float-batch/c rgb-image-batch-u8-or-float/c
                  rgb-triple/c))

(define padding-mode/c (or/c 'zeros 'border 'reflection))
(define scale/c (~> (real-in 0 1) (and/c positive?) ordered-pair/c))
(define ratio/c (ordered-pair/c (and/c real? positive?)))
(define strength/c (and/c real? (>=/c 0)))

(define (sides size)
  (if (list? size) size (list size size)))

(define (drawer x generator)
  (define n (car (shape x)))
  (define on (device x))
  (case-lambda
    [() (rand n #:device on #:generator generator)]
    [(lo hi)
     (define lo* (exact->inexact lo))
     (define hi* (exact->inexact hi))
     (+ lo* (* (- hi* lo*) (rand n #:device on #:generator generator)))]))

(define (coin x generator p)
  (reshape (< ((drawer x generator)) p) -1 1 1 1))

(define (mirrored sx flips)
  (if flips (* sx (where flips -1.0 1.0)) sx))

(define (cropped x size sx sy cx cy padding-mode)
  (match-define (list n c _ _) (shape x))
  (match-define (list oh ow) (sides size))
  (define zero (zeros-like sx))
  (define theta
    (~> (list (stack (list sx zero cx) 1) (stack (list zero sy cy) 1))
        (stack 1)
        (to-dtype (dtype x))))
  (grid-sample x (affine-grid theta (list n c oh ow))
               #:padding-mode padding-mode))

(define (blend-with x other factor)
  (blend x (other x) factor))

(define/contract-out (resized-crop x boxes size ;; noqa
                                   #:flip [flip #f]
                                   #:padding-mode [padding-mode 'reflection])
  (->i ([x float-batch/c] [boxes (x) (crop-boxes/c x)] [size image-size/c])
       (#:flip [flip (x) (or/c #f (per-sample-mask/c x))]
        #:padding-mode [padding-mode padding-mode/c])
       [result tensor?])
  (match-define (list _ _ h w) (shape x))
  (define (part k) (select boxes 1 k))
  (define-values (top left height width) (values (part 0) (part 1) (part 2) (part 3)))
  (define (centre start extent side)
    (~> start (* 2.0) (+ extent) (/ side) (- 1.0)))
  (cropped x size
           (mirrored (/ width w) flip) (/ height h)
           (centre left width w) (centre top height h)
           padding-mode))

(define/contract-out (adjust-brightness x factor) ;; noqa
  (->i ([x float-batch/c] [factor (x) (per-sample/c x)]) [result tensor?])
  (brightened x (column x factor)))

(define/contract-out (adjust-contrast x factor) ;; noqa
  (->i ([x rgb-float-batch/c] [factor (x) (per-sample/c x)]) [result tensor?])
  (blend x (contrast-mean x) (column x factor)))

(define/contract-out (adjust-saturation x factor) ;; noqa
  (->i ([x rgb-float-batch/c] [factor (x) (per-sample/c x)]) [result tensor?])
  (blend x (luma x) (column x factor)))

(define/contract-out (adjust-hue x factor) ;; noqa
  (->i ([x rgb-float-batch/c] [factor (x) (per-sample/c x (real-in -0.5 0.5))])
       [result tensor?])
  (hue-turned x (column x factor)))

(define/contract-out (rgb->grayscale x #:channels [channels 1]) ;; noqa
  (->* [rgb-float-batch/c] [#:channels (or/c 1 3)] tensor?)
  (define gray (luma x))
  (if (= channels 3) (broadcast-to gray (shape x)) gray))

(define/contract-out (solarize x threshold) ;; noqa
  (->i ([x float-batch/c] [threshold (x) (per-sample/c x)]) [result tensor?])
  (solarized x (column x threshold)))

(define/contract-out (random-affine-crop x size ;; noqa
                                         #:scale [scale '(0.08 1.0)]
                                         #:ratio [ratio '(3/4 4/3)]
                                         #:flip-p [flip-p 0.0]
                                         #:padding-mode [padding-mode 'reflection]
                                         #:generator [generator #f])
  (->i ([x float-batch/c] [size image-size/c])
       (#:scale [scale scale/c]
        #:ratio [ratio ratio/c]
        #:flip-p [flip-p (real-in 0 1)]
        #:padding-mode [padding-mode padding-mode/c]
        #:generator [generator (x) (generator-for/c x)])
       #:pre/name (x generator) "the generator lives on the batch's device"
       (generator-on? (supplied-or generator) (device x))
       [result tensor?])
  (match-define (list _ _ h w) (shape x))
  (define draw (drawer x generator))
  (define area (draw (car scale) (cadr scale)))
  (define ratio* (exp (draw (~> (car ratio) exact->inexact log)
                            (~> (cadr ratio) exact->inexact log))))
  (define (side a) (clamp (sqrt a) #:max 1.0))
  (define sx (side (* area ratio* (/ (exact->inexact h) w))))
  (define sy (side (* (/ area ratio*) (/ (exact->inexact w) h))))
  (define cx (* (- 1.0 sx) (draw -1.0 1.0)))
  (define cy (* (- 1.0 sy) (draw -1.0 1.0)))
  (define flips (and (positive? flip-p) (< (draw) flip-p)))
  (cropped x size (mirrored sx flips) sy cx cy padding-mode))

(define/contract-out (random-color-jitter x ;; noqa
                                          #:brightness [b 0]
                                          #:contrast [c 0]
                                          #:saturation [s 0]
                                          #:hue [h 0]
                                          #:p [p 1.0]
                                          #:generator [generator #f])
  (->i ([x rgb-float-batch/c])
       (#:brightness [b strength/c]
        #:contrast [c strength/c]
        #:saturation [s strength/c]
        #:hue [h (real-in 0 0.5)]
        #:p [p (real-in 0 1)]
        #:generator [generator (x) (generator-for/c x)])
       #:pre/name (x generator) "the generator lives on the batch's device"
       (generator-on? (supplied-or generator) (device x))
       [result tensor?])
  (cond
    [(zero? p) x]
    [else
     (define draw (drawer x generator))
     (define on (< (draw) p))
     (define (factor strength)
       (where on (draw (- 1.0 strength) (+ 1.0 strength)) 1.0))
     (define fb (factor b))
     (define fc (factor c))
     (define fs (factor s))
     (define fh (where on (draw (- h) h) 0.0))
     (~> x
         (brightened (column x fb))
         (blend-with contrast-mean (column x fc))
         (blend-with luma (column x fs))
         (hue-turned (column x fh)))]))

(define/contract-out (random-grayscale x #:p [p 0.1] #:generator [generator #f]) ;; noqa
  (->i ([x rgb-float-batch/c])
       (#:p [p (real-in 0 1)]
        #:generator [generator (x) (generator-for/c x)])
       #:pre/name (x generator) "the generator lives on the batch's device"
       (generator-on? (supplied-or generator) (device x))
       [result tensor?])
  (cond
    [(zero? p) x]
    [else (where (coin x generator p) (luma x) x)]))

(define/contract-out (random-solarize x threshold ;; noqa
                                      #:p [p 0.5]
                                      #:generator [generator #f])
  (->i ([x float-batch/c] [threshold real?])
       (#:p [p (real-in 0 1)]
        #:generator [generator (x) (generator-for/c x)])
       #:pre/name (x generator) "the generator lives on the batch's device"
       (generator-on? (supplied-or generator) (device x))
       [result tensor?])
  (cond
    [(zero? p) x]
    [else
     (where (coin x generator p) (solarized x (column x threshold)) x)]))

(struct crop-spec (size scale ratio flip-p jitter jitter-p gray-p solarize-p
                        mean std check)
  #:property prop:procedure
  (lambda (view images #:generator [generator #f])
    ((crop-spec-check view) images generator)
    (viewed view images generator)))

(define (normalized x vs op)
  (cond
    [vs (op x (reshape (tensor (map exact->inexact vs) #:dtype (dtype x)
                               #:device (device x))
                       1 3 1 1))]
    [else x]))

(define (viewed view images generator)
  (match-define (crop-spec size scale ratio flip-p (list b c s h) jitter-p
                           gray-p solarize-p mean std _)
    view)
  (define x
    (if (eq? (dtype images) 'uint8) (/ (to-dtype images 'float32) 255.0) images))
  (~> x
      (random-affine-crop size #:scale scale #:ratio ratio #:flip-p flip-p
                          #:generator generator)
      (random-color-jitter #:brightness b #:contrast c #:saturation s #:hue h
                           #:p jitter-p #:generator generator)
      (random-grayscale #:p gray-p #:generator generator)
      (random-solarize 0.5 #:p solarize-p #:generator generator)
      (normalized mean -)
      (normalized std /)))

(define view-call/c
  (->i ([images rgb-image-batch-u8-or-float/c]
        [generator (images) (generator-for/c images)])
       any))

(define/contract-out (crop-view? v) (-> any/c boolean?) ;; noqa
  (crop-spec? v))

(define/contract-out (crop-view #:size size ;; noqa
                                #:scale [scale '(0.08 1.0)]
                                #:ratio [ratio '(3/4 4/3)]
                                #:flip-p [flip-p 0.5]
                                #:jitter [jitter '(0.4 0.4 0.2 0.1)]
                                #:jitter-p [jitter-p 0.8]
                                #:gray-p [gray-p 0.2]
                                #:solarize-p [solarize-p 0.0]
                                #:mean [mean #f]
                                #:std [std #f])
  (->* [#:size image-size/c]
       [#:scale scale/c
        #:ratio ratio/c
        #:flip-p (real-in 0 1)
        #:jitter jitter-strengths/c
        #:jitter-p (real-in 0 1)
        #:gray-p (real-in 0 1)
        #:solarize-p (real-in 0 1)
        #:mean (or/c #f (rgb-triple/c))
        #:std (or/c #f (rgb-triple/c (and/c real? positive?)))]
       crop-view?)
  (crop-spec size scale ratio flip-p jitter jitter-p gray-p solarize-p mean std
             (contract view-call/c void 'crop-view 'caller 'crop-view #f)))

(define (one-size? views)
  (for/and ([v (in-list (cdr views))])
    (equal? (sides (crop-spec-size v)) (~> (car views) crop-spec-size sides))))

(define/contract-out (multi-crop images globals locals #:generator [generator #f]) ;; noqa
  (->i ([images rgb-image-batch-u8-or-float/c]
        [globals (non-empty-listof crop-view?)]
        [locals (listof crop-view?)])
       (#:generator [generator (images) (generator-for/c images)])
       #:pre/name (images generator) "the generator lives on the batch's device"
       (generator-on? (supplied-or generator) (device images))
       #:pre/name (globals) "global views share one #:size" (one-size? globals)
       #:pre/name (locals) "local views share one #:size"
       (or (null? locals) (one-size? locals))
       (values [global-views tensor?] [local-views (or/c #f tensor?)]))
  (define (group views)
    (for/list ([v (in-list views)])
      (v images #:generator generator)))
  (define big (cat (group globals) 0))
  (values big (and (pair? locals) (cat (group locals) 0))))
