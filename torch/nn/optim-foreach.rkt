#lang racket/base

(require (only-in racket/flonum flexpt)
         (only-in racket/list partition)
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

(define (decayed state name ps gs weight-decay)
  (if (zero? weight-decay)
      gs
      (~> (slots! state name ps)
          (g:foreach-copy! gs #f)
          (g:foreach-add-list! ps weight-decay))))

(define (momentum-buffers! state ps ds momentum)
  (define-values (held fresh)
    (partition (lambda (pd) (slot-ref state 'momentum-buffer (car pd)))
               (map cons ps ds)))
  (for ([pd (in-list fresh)])
    (define buf (slot! state 'momentum-buffer (car pd)))
    (copy! buf (cdr pd)))
  (unless (null? held)
    (~> (slots! state 'momentum-buffer (map car held))
        (g:foreach-mul-scalar! momentum)
        (g:foreach-add-list! (map cdr held) 1.0)))
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
               (g:foreach-copy! (slots! state 'scratch ps) gs #f)
               ds)
           (g:foreach-add-list! bufs momentum))]
      [else (momentum-buffers! state ps ds momentum)]))
  (g:foreach-add-list! ps steps (- lr)))

(define (sgd-fused! state ps gs #:lr lr #:momentum momentum
                    #:nesterov? nesterov? #:weight-decay weight-decay)
  (define (fused! pgs first?)
    (define qs (map car pgs))
    (define bufs
      (if (zero? momentum) '() (slots! state 'momentum-buffer qs)))
    (g:fused-sgd! qs (map cdr pgs) bufs weight-decay momentum lr 0.0
                  nesterov? #f first? #f #f))
  (define-values (held fresh)
    (partition (lambda (pg)
                 (or (zero? momentum)
                     (slot-ref state 'momentum-buffer (car pg))))
               (map cons ps gs)))
  (for ([pgs (in-list (list held fresh))]
        [first? (in-list '(#f #t))]
        #:when (pair? pgs))
    (fused! pgs first?)))

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
        (g:foreach-copy! vs #f)
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
  ((if decoupled? g:fused-adamw! g:fused-adam!)
   ps gs (slots! state 'exp-avg ps) (slots! state 'exp-avg-sq ps) '() steps
   lr beta1 beta2 weight-decay eps #f #f #f #f))

(define (rmsprop-foreach! state ps gs #:lr lr #:alpha alpha #:eps eps
                          #:weight-decay weight-decay #:momentum momentum)
  (define ds (decayed state 'decayed-gradient ps gs weight-decay))
  (define vs (slots! state 'square-avg ps))
  (g:foreach-mul-scalar! vs alpha)
  (g:foreach-addcmul-scalar! vs ds ds (- 1.0 alpha))
  (define avgs
    (~> (slots! state 'scratch ps)
        (g:foreach-copy! vs #f)
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
