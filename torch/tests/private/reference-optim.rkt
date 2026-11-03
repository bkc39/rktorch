#lang racket/base

(require (only-in racket/flonum flexpt)
         (only-in "../../foreign.rkt"
                  addcdiv! addcmul! copy! device dtype full mul! sqrt sub! to
                  with-no-grad zeros-like)
         (prefix-in g: (only-in "../../generated.rkt"
                                add-tensor add-tensor! div-scalar! lerp-scalar!)))

(provide reference-adam
         reference-rmsprop
         reference-sgd)

(define (moment-on table i p)
  (define m (hash-ref! table i (lambda () (zeros-like p))))
  (cond
    [(and (equal? (device m) (device p)) (eq? (dtype m) (dtype p))) m]
    [else
     (define moved (to m (device p) (dtype p)))
     (hash-set! table i moved)
     moved]))

(define (eps-on value p)
  (full value #:device (device p) #:dtype (dtype p)))

(define (decayed g p weight-decay)
  (if (zero? weight-decay) g (g:add-tensor g p (exact->inexact weight-decay))))

(define (reference-sgd #:lr lr
                       #:momentum [mu 0.0]
                       #:nesterov? [nesterov? #f]
                       #:weight-decay [weight-decay 0.0])
  (define bufs (make-hasheqv))
  (define (momentum-buffer! i p d)
    (cond
      [(hash-ref bufs i #f)
       (define buf (moment-on bufs i p))
       (mul! buf mu)
       (g:add-tensor! buf d 1.0)
       buf]
      [else
       (define fresh (zeros-like p))
       (copy! fresh d)
       (hash-set! bufs i fresh)
       fresh]))
  (lambda (ps gs)
    (with-no-grad
      (for ([p (in-list ps)] [g (in-list gs)] [i (in-naturals)])
        (define d (decayed g p weight-decay))
        (define step
          (cond
            [(zero? mu) d]
            [else
             (define buf (momentum-buffer! i p d))
             (if nesterov? (g:add-tensor d buf mu) buf)]))
        (g:add-tensor! p step (- lr))))))

(define (reference-adam #:lr [lr 1e-3]
                        #:beta1 [b1 0.9]
                        #:beta2 [b2 0.999]
                        #:eps [eps 1e-8]
                        #:weight-decay [weight-decay 0.0]
                        #:decoupled? [decoupled? #f])
  (define ms (make-hasheqv))
  (define vs (make-hasheqv))
  (define t 0)
  (lambda (ps gs)
    (set! t (add1 t))
    (define bc1 (- 1.0 (flexpt b1 (exact->inexact t))))
    (define bc2 (- 1.0 (flexpt b2 (exact->inexact t))))
    (with-no-grad
      (for ([p (in-list ps)] [g (in-list gs)] [i (in-naturals)])
        (when decoupled?
          (mul! p (- 1.0 (* lr weight-decay))))
        (define d (if decoupled? g (decayed g p weight-decay)))
        (define m (moment-on ms i p))
        (define v (moment-on vs i p))
        (g:lerp-scalar! m d (- 1.0 b1))
        (mul! v b2)
        (addcmul! v d d (- 1.0 b2))
        (define denom (sqrt v))
        (g:div-scalar! denom (flexpt bc2 0.5))
        (sub! denom (eps-on eps p) -1.0)
        (addcdiv! p m denom (- (/ lr bc1)))))))

(define (reference-rmsprop #:lr [lr 1e-2]
                           #:alpha [alpha 0.99]
                           #:eps [eps 1e-8]
                           #:weight-decay [weight-decay 0.0]
                           #:momentum [mu 0.0])
  (define sqs (make-hasheqv))
  (define bufs (make-hasheqv))
  (lambda (ps gs)
    (with-no-grad
      (for ([p (in-list ps)] [g (in-list gs)] [i (in-naturals)])
        (define d (decayed g p weight-decay))
        (define v (moment-on sqs i p))
        (mul! v alpha)
        (addcmul! v d d (- 1.0 alpha))
        (define avg (sqrt v))
        (sub! avg (eps-on eps p) -1.0)
        (cond
          [(zero? mu) (addcdiv! p d avg (- lr))]
          [else
           (define buf (moment-on bufs i p))
           (mul! buf mu)
           (addcdiv! buf d avg 1.0)
           (g:add-tensor! p buf (- lr))])))))
