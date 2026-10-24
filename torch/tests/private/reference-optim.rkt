#lang racket/base

;; e179ba7's optimizer steps, which read every buffer's placement at every
;; step, over explicit gradients; a test runs one beside the library's
(require (only-in "../../foreign.rkt"
                  add addcdiv! addcmul! copy! device dtype full mul mul! sqrt
                  sub! to with-no-grad zeros-like))

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
  (if (zero? weight-decay) g (add g (mul p weight-decay))))

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
       (sub! buf d -1.0)
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
             (if nesterov? (add d (mul buf mu)) buf)]))
        (sub! p step lr)))))

(define (reference-adam #:lr [lr 1e-3]
                        #:beta1 [b1 0.9]
                        #:beta2 [b2 0.999]
                        #:eps [eps 1e-8]
                        #:weight-decay [weight-decay 0.0])
  (define ms (make-hasheqv))
  (define vs (make-hasheqv))
  (define t 0)
  (lambda (ps gs)
    (set! t (add1 t))
    (define bc1 (- 1.0 (expt b1 t)))
    (define bc2 (- 1.0 (expt b2 t)))
    (with-no-grad
      (for ([p (in-list ps)] [g (in-list gs)] [i (in-naturals)])
        (define d (decayed g p weight-decay))
        (define m (moment-on ms i p))
        (define v (moment-on vs i p))
        (mul! m b1)
        (sub! m d (- b1 1.0))
        (mul! v b2)
        (addcmul! v d d (- 1.0 b2))
        (define denom (sqrt v))
        (mul! denom (/ 1.0 (expt bc2 0.5)))
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
           (sub! p buf lr)])))))
