#lang racket/base

(require (only-in racket/flonum flexpt)
         (only-in racket/function negate)
         (only-in threading ~>)
         (only-in "../foreign.rkt" copy! device zeros zeros-like)
         (prefix-in g: (only-in "../generated.rkt"
                                foreach-add-list! foreach-add-scalar!
                                foreach-addcdiv-scalar!
                                foreach-addcdiv-scalarlist!
                                foreach-addcmul-scalar! foreach-copy!
                                foreach-div-scalarlist! foreach-lerp-scalar!
                                foreach-mul-scalar! foreach-sqrt! fused-adam!
                                fused-adamw! fused-sgd!)))

(provide adam-foreach!
         adam-fused!
         bias-correction
         rmsprop-foreach!
         sgd-foreach!
         sgd-fused!
         slot!
         slot-ref
         step-count!)

(define (slot-table state name)
  (hash-ref! state name make-hasheq))

(define (slot-ref state name p)
  (hash-ref (slot-table state name) p #f))

(define (slot-set! state name p v)
  (hash-set! (slot-table state name) p v))

(define (slot! state name p [fresh zeros-like])
  (hash-ref! (slot-table state name) p (lambda () (fresh p))))

(define (slots! state name ps [fresh zeros-like])
  (for/list ([p (in-list ps)])
    (slot! state name p fresh)))

(define (step-count! state p)
  (define t (add1 (or (slot-ref state 'step p) 0)))
  (slot-set! state 'step p t)
  t)

(define (bias-correction beta t)
  (- 1.0 (flexpt beta (exact->inexact t))))

(define (step-tensor p)
  (zeros #:device (device p) #:dtype 'float32))

(define (foreach-copy! dsts srcs #:non-blocking? [non-blocking? #f])
  (g:foreach-copy! dsts srcs non-blocking?))

(define (fused-sgd! ps gs bufs #:lr lr #:momentum momentum
                    #:weight-decay weight-decay #:nesterov? nesterov?
                    #:first-step? first-step? #:dampening [dampening 0.0]
                    #:maximize? [maximize? #f])
  (g:fused-sgd! ps gs bufs weight-decay momentum lr dampening nesterov?
                maximize? first-step? #f #f))

(define (fused-adam! ps gs ms vs steps #:decoupled? decoupled? #:lr lr
                     #:beta1 beta1 #:beta2 beta2 #:weight-decay weight-decay
                     #:eps eps #:amsgrad? [amsgrad? #f] #:maximize? [maximize? #f])
  ((if decoupled? g:fused-adamw! g:fused-adam!)
   ps gs ms vs '() steps lr beta1 beta2 weight-decay eps amsgrad? maximize?
   #f #f))

(define (pairs-where keep? ps xs)
  (for/lists (qs ys) ([p (in-list ps)] [x (in-list xs)] #:when (keep? p))
    (values p x)))

(define (decayed state name ps gs weight-decay)
  (if (zero? weight-decay)
      gs
      (~> (slots! state name ps)
          (foreach-copy! gs)
          (g:foreach-add-list! ps weight-decay))))

(define (momentum-buffers! state ps ds momentum)
  (define (held? p) (slot-ref state 'momentum-buffer p))
  (define-values (held-ps held-ds) (pairs-where held? ps ds))
  (define-values (fresh-ps fresh-ds) (pairs-where (negate held?) ps ds))
  (for ([p (in-list fresh-ps)] [d (in-list fresh-ds)])
    (copy! (slot! state 'momentum-buffer p) d))
  (unless (null? held-ps)
    (~> (slots! state 'momentum-buffer held-ps)
        (g:foreach-mul-scalar! momentum)
        (g:foreach-add-list! held-ds 1.0)))
  (slots! state 'momentum-buffer ps))

(define (sgd-foreach! state ps gs #:lr lr #:momentum momentum
                      #:nesterov? nesterov? #:weight-decay weight-decay)
  (define ds (decayed state 'scratch ps gs weight-decay))
  (define steps
    (cond
      [(zero? momentum) ds]
      [nesterov?
       (define bufs (momentum-buffers! state ps ds momentum))
       (~> (if (eq? ds gs)
               (foreach-copy! (slots! state 'scratch ps) gs)
               ds)
           (g:foreach-add-list! bufs momentum))]
      [else (momentum-buffers! state ps ds momentum)]))
  (g:foreach-add-list! ps steps (- lr)))

(define (sgd-fused! state ps gs #:lr lr #:momentum momentum
                    #:nesterov? nesterov? #:weight-decay weight-decay)
  (define (fused! qs hs #:first-step? first-step?)
    (define bufs
      (if (zero? momentum) '() (slots! state 'momentum-buffer qs)))
    (fused-sgd! qs hs bufs #:lr lr #:momentum momentum
                #:weight-decay weight-decay #:nesterov? nesterov?
                #:first-step? first-step?))
  (define (held? p)
    (or (zero? momentum) (slot-ref state 'momentum-buffer p)))
  (define-values (held-ps held-gs) (pairs-where held? ps gs))
  (define-values (fresh-ps fresh-gs) (pairs-where (negate held?) ps gs))
  (unless (null? held-ps)
    (fused! held-ps held-gs #:first-step? #f))
  (unless (null? fresh-ps)
    (fused! fresh-ps fresh-gs #:first-step? #t)))

(define (adam-foreach! state ps gs #:lr lr #:beta1 beta1 #:beta2 beta2
                       #:eps eps #:weight-decay weight-decay
                       #:decoupled? decoupled?)
  (define steps (map (lambda (p) (step-count! state p)) ps))
  (define ds
    (cond
      [decoupled?
       (g:foreach-mul-scalar! ps (- 1.0 (* lr weight-decay)))
       gs]
      [else (decayed state 'scratch ps gs weight-decay)]))
  (define ms (slots! state 'exp-avg ps))
  (define vs (slots! state 'exp-avg-sq ps))
  (g:foreach-lerp-scalar! ms ds (- 1.0 beta1))
  (g:foreach-mul-scalar! vs beta2)
  (g:foreach-addcmul-scalar! vs ds ds (- 1.0 beta2))
  (define denoms
    (~> (slots! state 'scratch ps)
        (foreach-copy! vs)
        g:foreach-sqrt!
        (g:foreach-div-scalarlist!
         (map (lambda (t) (flexpt (bias-correction beta2 t) 0.5)) steps))
        (g:foreach-add-scalar! eps)))
  (g:foreach-addcdiv-scalarlist!
   ps ms denoms
   (map (lambda (t) (* (/ lr (bias-correction beta1 t)) -1.0)) steps)))

(define (adam-fused! state ps gs #:lr lr #:beta1 beta1 #:beta2 beta2
                     #:eps eps #:weight-decay weight-decay
                     #:decoupled? decoupled?)
  (define steps (g:foreach-add-scalar! (slots! state 'step ps step-tensor) 1.0))
  (fused-adam! ps gs (slots! state 'exp-avg ps) (slots! state 'exp-avg-sq ps)
               steps #:decoupled? decoupled? #:lr lr #:beta1 beta1
               #:beta2 beta2 #:weight-decay weight-decay #:eps eps))

(define (rmsprop-foreach! state ps gs #:lr lr #:alpha alpha #:eps eps
                          #:weight-decay weight-decay #:momentum momentum)
  (define ds (decayed state 'decayed-gradient ps gs weight-decay))
  (define vs (slots! state 'square-avg ps))
  (g:foreach-mul-scalar! vs alpha)
  (g:foreach-addcmul-scalar! vs ds ds (- 1.0 alpha))
  (define avgs
    (~> (slots! state 'scratch ps)
        (foreach-copy! vs)
        g:foreach-sqrt!
        (g:foreach-add-scalar! eps)))
  (cond
    [(zero? momentum) (g:foreach-addcdiv-scalar! ps ds avgs (- lr))]
    [else
     (define bufs
       (~> (slots! state 'momentum-buffer ps)
           (g:foreach-mul-scalar! momentum)
           (g:foreach-addcdiv-scalar! ds avgs 1.0)))
     (g:foreach-add-list! ps bufs (- lr))]))
