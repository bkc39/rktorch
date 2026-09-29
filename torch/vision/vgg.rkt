#lang racket/base

(require (only-in racket/contract/base ->*)
         (only-in "../foreign.rkt" relu)
         (only-in "../foreign/contracts.rkt" rgb-image-batch/c)
         (only-in "../nn/conv.rkt" Conv2d MaxPool2d)
         (only-in "../nn/layer.rkt" define-layer)
         (only-in "../nn/sequential.rkt" Sequential)
         (only-in "../nn/state-dict.rkt" load-state!)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "weights.rkt" pretrained-weights))

(define vgg16-plan
  '(64 64 pool 128 128 pool 256 256 256 pool 512 512 512 pool 512 512 512 pool))

(define (feature-steps plan)
  (for/fold ([in 3] [steps '()] #:result (reverse steps))
            ([width (in-list plan)])
    (if (eq? width 'pool)
        (values in (cons (MaxPool2d 2 #:stride 2) steps))
        (values width (list* relu (Conv2d in width 3 #:padding 1) steps)))))

(define-layer VGG16Features (features) ;; noqa
  #:predicate vgg16-features?
  #:contract (->* [] [] vgg16-features?)
  #:init ()
  (set! features (Sequential (feature-steps vgg16-plan)))
  #:forward ([x : rgb-image-batch/c])
  (features x))

(define/contract-out (vgg16-features #:pretrained? [pretrained? #f]) ;; noqa
  (->* [] [#:pretrained? boolean?] vgg16-features?)
  (define net (VGG16Features))
  (when pretrained?
    (load-state! net (pretrained-weights 'vgg16-features-imagenet1k-v1)))
  net)
