#lang racket/base

(require (only-in racket/contract/base
                  -> ->i </c >=/c and/c any contract flat-contract-predicate
                  flat-named-contract list/c or/c)
         (only-in racket/match match-define)
         (only-in "../foreign.rkt" dtype shape)
         (only-in "../foreign/contracts.rkt" image-batch/c)
         (only-in "../foreign/ops.rkt" any-float-dtype/c)
         (only-in "../nn/layer.rkt" define-layer)
         (only-in "../private/contract.rkt" define/contract-out supplied-or))

(define image-size/c
  (flat-named-contract
   'image-size
   (or/c exact-positive-integer?
         (list/c exact-positive-integer? exact-positive-integer?))))

(define output/c (or/c 'head 'cls 'tokens 'attention))

(define float-dtype? (flat-contract-predicate any-float-dtype/c))

(define (sides image-size)
  (if (list? image-size) image-size (list image-size image-size)))

(define (image-fits x in-channels patch)
  (match-define (list _ channels h w) (shape x))
  (cond
    [(not (float-dtype? (dtype x)))
     (format "x is a floating-point batch, not ~a" (dtype x))]
    [(not (= channels in-channels))
     (format "x has ~a channels where the model reads ~a" channels in-channels)]
    [(not (and (positive? h) (positive? w)
               (zero? (remainder h patch)) (zero? (remainder w patch))))
     (format "each side of x is a positive multiple of the patch ~a, not ~a by ~a"
             patch h w)]
    [else #t]))

(define (patch-call/c in-channels patch)
  (->i ([x image-batch/c])
       #:pre/desc (x) (image-fits x in-channels patch)
       any))

(define (vit-call/c in-channels patch)
  (->i ([x image-batch/c] [output output/c])
       #:pre/desc (x) (image-fits x in-channels patch)
       any))

(module+ private
  (provide positions)) ;; noqa

(define (positions pos-embed grid target)
  (raise (exn:fail:unsupported "positions: not implemented"
                               (current-continuation-marks))))

(define (checker who call/c)
  (contract call/c void who 'caller who #f))

(define-layer PatchEmbed (proj patch check) ;; noqa
  #:predicate patch-embed?
  #:contract (-> exact-positive-integer? exact-positive-integer?
                 exact-positive-integer? patch-embed?)
  #:init (in-channels width patch-side)
  (set! patch patch-side)
  (set! check (checker 'PatchEmbed (patch-call/c in-channels patch-side)))
  (raise (exn:fail:unsupported "PatchEmbed: not implemented"
                               (current-continuation-marks)))
  #:forward (x)
  (check x)
  x)

(define-layer VisionTransformer ;; noqa
  (patch-embed cls-token pos-embed blocks norm head patch grid check)
  #:predicate vision-transformer?
  #:contract
  (->i ()
       (#:image-size [image-size image-size/c]
        #:patch [patch exact-positive-integer?]
        #:in-channels [in-channels exact-positive-integer?]
        #:width [width exact-positive-integer?]
        #:depth [depth exact-positive-integer?]
        #:heads [heads exact-positive-integer?]
        #:ffn-width [ffn-width exact-positive-integer?]
        #:dropout [dropout (and/c real? (>=/c 0) (</c 1))]
        #:layer-norm-eps [eps (and/c real? positive?)]
        #:classes [classes (or/c #f exact-positive-integer?)])
       #:pre/name (image-size patch) "#:patch divides each side of #:image-size"
       (let ([p (supplied-or patch 16)])
         (for/and ([side (in-list (sides (supplied-or image-size 224)))])
           (zero? (remainder side p))))
       #:pre/name (width heads) "#:heads divides #:width"
       (zero? (remainder (supplied-or width 768) (supplied-or heads 12)))
       [_ vision-transformer?])
  #:init (#:image-size [image-size 224] #:patch [patch-side 16]
          #:in-channels [in-channels 3]
          #:width [width 768] #:depth [depth 12] #:heads [heads 12]
          #:ffn-width [ffn-width (* 4 width)] #:dropout [dropout 0.0]
          #:layer-norm-eps [eps 1e-6] #:classes [classes #f])
  (set! patch patch-side)
  (set! check (checker 'VisionTransformer (vit-call/c in-channels patch-side)))
  (raise (exn:fail:unsupported "VisionTransformer: not implemented"
                               (current-continuation-marks)))
  #:forward (x #:output [output 'head])
  (check x output)
  x)

(define/contract-out (huggingface-vit-key key) ;; noqa
  (-> string? (or/c string? #f))
  (raise (exn:fail:unsupported "huggingface-vit-key: not implemented"
                               (current-continuation-marks))))
