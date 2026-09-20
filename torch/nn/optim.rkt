#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i >=/c >/c any/c contract-out listof real-in
                  unsupplied-arg?)
         (only-in racket/generic define-generics)
         (only-in "../foreign.rkt"
                  add addcdiv! addcmul! copy! full maybe-grad mul mul! sqrt
                  sub! tensor-device tensor-dtype tensor? to with-no-grad
                  zero-grad! zeros-like)
         (only-in "../private/contract.rkt" define/contract-out))

(provide gen:optimizer
         optimizer?
         optimizer-step!
         optimizer-parameters
         optimizer-lr
         optimizer-set-lr!
         sgd-lr
         (contract-out [sgd? (-> any/c boolean?)]
                       [adam? (-> any/c boolean?)]
                       [rmsprop? (-> any/c boolean?)]))

(module+ checked
  (provide (contract-out [optimizer? (-> any/c boolean?)])))

(define-generics optimizer
  (optimizer-step! optimizer)
  (optimizer-parameters optimizer)
  (optimizer-lr optimizer)
  (optimizer-set-lr! optimizer lr))

;; a moment or momentum buffer lives where its parameter does, and follows
;; it after a move
(define (moment-on table p)
  (define m (hash-ref! table p (lambda () (zeros-like p))))
  (define dev (tensor-device p))
  (define dt (tensor-dtype p))
  (cond
    [(and (equal? (tensor-device m) dev) (eq? (tensor-dtype m) dt)) m]
    [else
     (define moved (to m dev dt))
     (hash-set! table p moved)
     moved]))

(define (scalar-on table value p)
  (define dev (tensor-device p))
  (define dt (tensor-dtype p))
  (define (fresh) (full value #:device dev #:dtype dt))
  (define s (hash-ref! table p fresh))
  (cond
    [(and (equal? (tensor-device s) dev) (eq? (tensor-dtype s) dt)) s]
    [else
     (define moved (fresh))
     (hash-set! table p moved)
     moved]))

;; torch's L2 weight decay: the gradient the update sees is g + wd * p
(define (decayed g p weight-decay)
  (if (zero? weight-decay) g (add g (mul p weight-decay))))

(struct sgd (params [lr #:mutable] momentum nesterov? weight-decay bufs)
  #:constructor-name make-sgd
  #:name sgd-optimizer ;; noqa
  #:methods gen:optimizer
  [(define (optimizer-parameters opt) (sgd-params opt))
   (define (optimizer-lr opt) (sgd-lr opt))
   (define (optimizer-set-lr! opt lr) (set-sgd-lr! opt lr))
   (define (optimizer-step! opt) (sgd-do-step! opt))])

(define/contract-out (sgd params ;; noqa
                          #:lr lr
                          #:momentum [momentum 0.0]
                          #:nesterov? [nesterov? #f]
                          #:weight-decay [weight-decay 0.0])
  (->i ([params (listof tensor?)] #:lr [lr (>=/c 0)])
       (#:momentum [momentum (real-in 0 1)]
        #:nesterov? [nesterov? boolean?]
        #:weight-decay [weight-decay (>=/c 0)])
       #:pre/name (momentum nesterov?)
       "Nesterov momentum requires a momentum"
       (or (unsupplied-arg? nesterov?)
           (not nesterov?)
           (and (not (unsupplied-arg? momentum)) (> momentum 0)))
       [result sgd?])
  (make-sgd params lr momentum nesterov? weight-decay (make-hasheq)))

;; torch.optim.SGD: the first step copies the gradient into the buffer, the
;; later ones blend it in; Nesterov looks one blend ahead
(define (momentum-buffer! opt p d)
  (define table (sgd-bufs opt))
  (cond
    [(hash-ref table p #f)
     (define buf (moment-on table p))
     (mul! buf (sgd-momentum opt))
     (sub! buf d -1.0)
     buf]
    [else
     (define fresh (zeros-like p))
     (copy! fresh d)
     (hash-set! table p fresh)
     fresh]))

(define (sgd-do-step! opt)
  (with-no-grad
    (define mu (sgd-momentum opt))
    (for ([p (in-list (sgd-params opt))])
      (define g (maybe-grad p))
      (when g
        (define d (decayed g p (sgd-weight-decay opt)))
        (define step
          (cond
            [(zero? mu) d]
            [else
             (define buf (momentum-buffer! opt p d))
             (if (sgd-nesterov? opt) (add d (mul buf mu)) buf)]))
        (sub! p step (sgd-lr opt))))))

(struct adam (params [lr #:mutable] beta1 beta2 eps weight-decay step-box
                     m v scalars)
  #:constructor-name make-adam
  #:name adam-optimizer ;; noqa
  #:methods gen:optimizer
  [(define (optimizer-parameters opt) (adam-params opt))
   (define (optimizer-lr opt) (adam-lr opt))
   (define (optimizer-set-lr! opt lr) (set-adam-lr! opt lr))
   (define (optimizer-step! opt) (adam-do-step! opt))])

(define/contract-out (adam params ;; noqa
                           #:lr [lr 1e-3]
                           #:beta1 [beta1 0.9]
                           #:beta2 [beta2 0.999]
                           #:eps [eps 1e-8]
                           #:weight-decay [weight-decay 0.0])
  (->* [(listof tensor?)]
       [#:lr (>=/c 0) #:beta1 real? #:beta2 real? #:eps real?
        #:weight-decay (>=/c 0)]
       adam?)
  (make-adam params lr beta1 beta2 eps weight-decay (box 0)
             (make-hasheq) (make-hasheq) (make-hasheq)))

(define (adam-do-step! opt)
  (with-no-grad
    (set-box! (adam-step-box opt) (add1 (unbox (adam-step-box opt))))
    (define t (unbox (adam-step-box opt)))
    (define b1 (adam-beta1 opt))
    (define b2 (adam-beta2 opt))
    (define lr (adam-lr opt))
    (define eps (adam-eps opt))
    (define bc1 (- 1.0 (expt b1 t)))
    (define bc2 (- 1.0 (expt b2 t)))
    (for ([p (in-list (adam-params opt))])
      (define g0 (maybe-grad p))
      (when g0
        (define g (decayed g0 p (adam-weight-decay opt)))
        (define m (moment-on (adam-m opt) p))
        (define v (moment-on (adam-v opt) p))
        (mul! m b1)
        (sub! m g (- b1 1.0))
        (mul! v b2)
        (addcmul! v g g (- 1.0 b2))
        (define denom (sqrt v))
        (mul! denom (/ 1.0 (expt bc2 0.5)))
        (sub! denom (scalar-on (adam-scalars opt) eps p) -1.0)
        (addcdiv! p m denom (- (/ lr bc1)))))))

(struct rmsprop (params [lr #:mutable] alpha eps weight-decay momentum
                        sq bufs scalars)
  #:constructor-name make-rmsprop
  #:name rmsprop-optimizer ;; noqa
  #:methods gen:optimizer
  [(define (optimizer-parameters opt) (rmsprop-params opt))
   (define (optimizer-lr opt) (rmsprop-lr opt))
   (define (optimizer-set-lr! opt lr) (set-rmsprop-lr! opt lr))
   (define (optimizer-step! opt) (rmsprop-do-step! opt))])

(define/contract-out (rmsprop params ;; noqa
                              #:lr [lr 1e-2]
                              #:alpha [alpha 0.99]
                              #:eps [eps 1e-8]
                              #:weight-decay [weight-decay 0.0]
                              #:momentum [momentum 0.0])
  (->* [(listof tensor?)]
       [#:lr (>=/c 0) #:alpha (real-in 0 1) #:eps (>/c 0)
        #:weight-decay (>=/c 0)
        #:momentum (real-in 0 1)]
       rmsprop?)
  (make-rmsprop params lr alpha eps weight-decay momentum
                (make-hasheq) (make-hasheq) (make-hasheq)))

;; torch.optim.RMSprop, uncentered: the running square average, its root
;; plus eps as the divisor, and a zero-initialised momentum buffer when asked
(define (rmsprop-do-step! opt)
  (with-no-grad
    (define alpha (rmsprop-alpha opt))
    (define mu (rmsprop-momentum opt))
    (define lr (rmsprop-lr opt))
    (for ([p (in-list (rmsprop-params opt))])
      (define g0 (maybe-grad p))
      (when g0
        (define g (decayed g0 p (rmsprop-weight-decay opt)))
        (define v (moment-on (rmsprop-sq opt) p))
        (mul! v alpha)
        (addcmul! v g g (- 1.0 alpha))
        (define avg (sqrt v))
        (sub! avg (scalar-on (rmsprop-scalars opt) (rmsprop-eps opt) p) -1.0)
        (cond
          [(zero? mu) (addcdiv! p g avg (- lr))]
          [else
           (define buf (moment-on (rmsprop-bufs opt) p))
           (mul! buf mu)
           (addcdiv! buf g avg 1.0)
           (sub! p buf lr)])))))

(define/contract-out (step! opt) ;; noqa
  (-> optimizer? void?)
  (optimizer-step! opt))

(define/contract-out (zero-grads! opt) ;; noqa
  (-> optimizer? void?)
  (for-each zero-grad! (optimizer-parameters opt)))

(define/contract-out (learning-rate opt) ;; noqa
  (-> optimizer? real?)
  (optimizer-lr opt))

(define/contract-out (set-learning-rate! opt lr) ;; noqa
  (-> optimizer? real? void?)
  (optimizer-set-lr! opt lr))
