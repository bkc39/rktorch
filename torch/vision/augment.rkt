#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i >=/c and/c any any/c contract non-empty-listof
                  listof or/c real-in)
         (only-in "../foreign.rkt" device tensor?)
         (only-in "../private/contract.rkt" define/contract-out supplied-or)
         (only-in "../foreign/generator.rkt" generator-on?)
         (only-in "private/contracts.rkt"
                  crop-boxes/c float-batch/c generator-for/c image-size/c
                  jitter-strengths/c ordered-pair/c per-sample-mask/c
                  per-sample/c rgb-float-batch/c rgb-image-batch-u8-or-float/c
                  rgb-triple/c))

(define padding-mode/c (or/c 'zeros 'border 'reflection))
(define scale/c (ordered-pair/c (and/c (real-in 0 1) positive?)))
(define ratio/c (ordered-pair/c (and/c real? positive?)))
(define strength/c (and/c real? (>=/c 0)))

(define/contract-out (resized-crop x boxes size ;; noqa
                                   #:flip [flip #f]
                                   #:padding-mode [padding-mode 'reflection])
  (->i ([x float-batch/c] [boxes (x) (crop-boxes/c x)] [size image-size/c])
       (#:flip [flip (x) (or/c #f (per-sample-mask/c x))]
        #:padding-mode [padding-mode padding-mode/c])
       [result tensor?])
  (error 'unimplemented))

(define/contract-out (adjust-brightness x factor) ;; noqa
  (->i ([x float-batch/c] [factor (x) (per-sample/c x)]) [result tensor?])
  (error 'unimplemented))

(define/contract-out (adjust-contrast x factor) ;; noqa
  (->i ([x rgb-float-batch/c] [factor (x) (per-sample/c x)]) [result tensor?])
  (error 'unimplemented))

(define/contract-out (adjust-saturation x factor) ;; noqa
  (->i ([x rgb-float-batch/c] [factor (x) (per-sample/c x)]) [result tensor?])
  (error 'unimplemented))

(define/contract-out (adjust-hue x factor) ;; noqa
  (->i ([x rgb-float-batch/c] [factor (x) (per-sample/c x (real-in -0.5 0.5))])
       [result tensor?])
  (error 'unimplemented))

(define/contract-out (rgb->grayscale x #:channels [channels 1]) ;; noqa
  (->* [rgb-float-batch/c] [#:channels (or/c 1 3)] tensor?)
  (error 'unimplemented))

(define/contract-out (solarize x threshold) ;; noqa
  (->i ([x float-batch/c] [threshold (x) (per-sample/c x)]) [result tensor?])
  (error 'unimplemented))

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
  (error 'unimplemented))

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
  (error 'unimplemented))

(define/contract-out (random-grayscale x #:p [p 0.1] #:generator [generator #f]) ;; noqa
  (->i ([x rgb-float-batch/c])
       (#:p [p (real-in 0 1)]
        #:generator [generator (x) (generator-for/c x)])
       #:pre/name (x generator) "the generator lives on the batch's device"
       (generator-on? (supplied-or generator) (device x))
       [result tensor?])
  (error 'unimplemented))

(define/contract-out (random-solarize x threshold ;; noqa
                                      #:p [p 0.5]
                                      #:generator [generator #f])
  (->i ([x float-batch/c] [threshold real?])
       (#:p [p (real-in 0 1)]
        #:generator [generator (x) (generator-for/c x)])
       #:pre/name (x generator) "the generator lives on the batch's device"
       (generator-on? (supplied-or generator) (device x))
       [result tensor?])
  (error 'unimplemented))

(struct crop-spec (size scale ratio flip-p jitter jitter-p gray-p solarize-p
                        mean std check)
  #:property prop:procedure
  (lambda (view images #:generator [generator #f])
    ((crop-spec-check view) images generator)
    (error 'unimplemented)))

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

(define (sides size)
  (if (list? size) size (list size size)))

(define (one-size? views)
  (for/and ([v (in-list (cdr views))])
    (equal? (sides (crop-spec-size v)) (sides (crop-spec-size (car views))))))

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
  (error 'unimplemented))
