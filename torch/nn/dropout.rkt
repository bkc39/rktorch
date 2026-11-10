#lang racket/base

(require (only-in racket/contract/base ->* <=/c >=/c and/c flat-contract-predicate or/c)
         (only-in threading ~>)
         (only-in "../foreign.rkt"
                  * / < current-generator device dtype generator? generator-device
                  rand-like tensor? to-dtype)
         (only-in "../foreign/generator.rkt" generator-on?)
         (only-in "../foreign/ops.rkt" any-float-dtype/c)
         (prefix-in g: (only-in "../generated.rkt" dropout))
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "layer.rkt" define-layer training? with-mode))

(define default-p 0.5)

(define probability/c (and/c real? (>=/c 0) (<=/c 1)))

(define float-dtype? (flat-contract-predicate any-float-dtype/c))

(define (draw-dtype type)
  (if (memq type '(float16 bfloat16)) 'float32 type))

(define (check-source who x source)
  (cond
    [(not (generator? source))
     (raise-argument-error who "(or/c generator? #f) as #:generator" source)]
    [(not (generator-on? source (device x)))
     (raise-arguments-error who "contract violation; the generator lives on the tensor's device"
                            "generator device" (generator-device source)
                            "tensor device" (device x))]
    [(not (float-dtype? (dtype x)))
     (raise-argument-error who "a floating-point tensor when drawing from a generator" x)]
    [else (void)]))

(define (masked who x rate source)
  (check-source who x source)
  (define keep (- 1.0 rate))
  (* x (~> (rand-like x #:dtype (draw-dtype (dtype x)) #:generator source)
           (< keep)
           (to-dtype (dtype x))
           (/ keep))))

(define (drop who x p training? generator)
  (define rate (exact->inexact p))
  (define source
    (and training? (< 0.0 rate 1.0) (or generator (current-generator))))
  (if source
      (masked who x rate source)
      (g:dropout x rate training?)))

(define/contract-out dropout ;; noqa
  (->* [tensor?]
       [#:p probability/c
        #:training? boolean?
        #:generator (or/c generator? #f)]
       tensor?)
  (lambda (x #:p [p default-p] #:training? [training? #t] #:generator [generator #f])
    (drop 'dropout x p training? generator)))

(define-layer Dropout (p) ;; noqa
  #:contract (->* [] [#:p probability/c] dropout?)
  #:init (#:p [p default-p])
  #:forward (x #:generator [generator #f])
  (with-mode (drop 'Dropout x p (training? mode) generator)))
