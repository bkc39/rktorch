#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i >=/c any/c contract-out flat-named-contract listof
                  unsupplied-arg?)
         (only-in racket/generic define-generics)
         (only-in "../foreign.rkt"
                  add addcdiv! addcmul! copy! cpu-device device dtype full mul
                  mul! sqrt sub! tensor? to with-no-grad zero-grad! zeros-like)
         (only-in "../foreign/autograd-ops.rkt" reset-grads! with-gradient)
         (only-in "../foreign/ops.rkt" move-generation)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "../private/step-log.rkt" log-optimizer-step!))

(provide gen:optimizer
         optimizer?
         optimizer-step!
         optimizer-parameters
         optimizer-lr
         optimizer-set-lr!
         sgd-lr
         (contract-out [sgd? (-> any/c boolean?)]
                       [adam? (-> any/c boolean?)]
                       [adamw? (-> any/c boolean?)]
                       [rmsprop? (-> any/c boolean?)]))

(module+ checked
  (provide (contract-out [optimizer? (-> any/c boolean?)])))

(module+ private
  (provide optimizer-state))

(define-generics optimizer
  (optimizer-step! optimizer)
  (optimizer-parameters optimizer)
  (optimizer-lr optimizer)
  (optimizer-set-lr! optimizer lr))

(struct tracked (tables [generation #:mutable]))

(define (optimizer-state opt)
  (for*/list ([table (in-list (tracked-tables opt))]
              [s (in-hash-values table)])
    s))

(define (placed-as s p)
  (define dev (device p))
  (define dt (dtype p))
  (if (and (equal? (device s) dev) (eq? (dtype s) dt)) s (to s dev dt)))

(define (follow-moves! opt)
  (define now (move-generation))
  (cond
    [(eqv? (tracked-generation opt) now) (void)]
    [else
     (for* ([table (in-list (tracked-tables opt))]
            [p (in-list (hash-keys table))])
       (hash-update! table p (lambda (s) (placed-as s p))))
     (set-tracked-generation! opt now)]))

(define (unit-scalar)
  (full 1.0 #:device (cpu-device) #:dtype 'float32))

(define (decayed g p weight-decay)
  (if (zero? weight-decay) g (add g (mul p weight-decay))))

(define floating-tensor/c
  (flat-named-contract 'floating-tensor/c
                       (lambda (v) (and (tensor? v)
                                        (memq (dtype v) '(float16 bfloat16 float32 float64))
                                        #t))))

(define (fused-parameters/c fused?)
  (if (eq? fused? #t) (listof floating-tensor/c) (listof tensor?)))

(struct sgd tracked (params [lr #:mutable] momentum nesterov? weight-decay
                            foreach? fused? bufs)
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
                          #:weight-decay [weight-decay 0.0]
                          #:foreach? [foreach? #t]
                          #:fused? [fused? #f])
  (->i ([params (fused?) (fused-parameters/c fused?)] #:lr [lr (>=/c 0)])
       (#:momentum [momentum (>=/c 0)]
        #:nesterov? [nesterov? boolean?]
        #:weight-decay [weight-decay (>=/c 0)]
        #:foreach? [foreach? boolean?]
        #:fused? [fused? boolean?])
       #:pre/name (momentum nesterov?)
       "Nesterov momentum requires a momentum"
       (or (unsupplied-arg? nesterov?)
           (not nesterov?)
           (and (not (unsupplied-arg? momentum)) (> momentum 0)))
       #:pre/name (foreach? fused?)
       "fused and foreach cannot both be requested"
       (not (and (eq? foreach? #t) (eq? fused? #t)))
       [result sgd?])
  (define bufs (make-hasheq))
  (make-sgd (list bufs) #f params lr momentum nesterov? weight-decay
            foreach? fused? bufs))

(define (momentum-buffer! opt p d)
  (define table (sgd-bufs opt))
  (cond
    [(hash-ref table p #f)
     => (lambda (buf)
          (mul! buf (sgd-momentum opt))
          (sub! buf d -1.0)
          buf)]
    [else
     (define fresh (zeros-like p))
     (copy! fresh d)
     (hash-set! table p fresh)
     fresh]))

(define (sgd-do-step! opt)
  (with-no-grad
    (define mu (sgd-momentum opt))
    (follow-moves! opt)
    (for ([p (in-list (sgd-params opt))])
      (with-gradient (g p)
        (define d (decayed g p (sgd-weight-decay opt)))
        (define step
          (cond
            [(zero? mu) d]
            [else
             (define buf (momentum-buffer! opt p d))
             (if (sgd-nesterov? opt) (add d (mul buf mu)) buf)]))
        (sub! p step (sgd-lr opt))))))

(struct adam tracked (params [lr #:mutable] beta1 beta2 eps weight-decay
                             foreach? fused? steps m v one)
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
                           #:weight-decay [weight-decay 0.0]
                           #:foreach? [foreach? #t]
                           #:fused? [fused? #f])
  (->i ([params (fused?) (fused-parameters/c fused?)])
       (#:lr [lr (>=/c 0)] #:beta1 [beta1 real?] #:beta2 [beta2 real?]
        #:eps [eps real?] #:weight-decay [weight-decay (>=/c 0)]
        #:foreach? [foreach? boolean?] #:fused? [fused? boolean?])
       #:pre/name (foreach? fused?)
       "fused and foreach cannot both be requested"
       (not (and (eq? foreach? #t) (eq? fused? #t)))
       [result adam?])
  (define m (make-hasheq))
  (define v (make-hasheq))
  (make-adam (list m v) #f params lr beta1 beta2 eps weight-decay
             foreach? fused? (make-hasheq) m v (unit-scalar)))

(struct adamw adam-optimizer ()
  #:constructor-name make-adamw
  #:name adamw-optimizer ;; noqa
  #:omit-define-syntaxes)

(define/contract-out (adamw params ;; noqa
                            #:lr [lr 1e-3]
                            #:beta1 [beta1 0.9]
                            #:beta2 [beta2 0.999]
                            #:eps [eps 1e-8]
                            #:weight-decay [weight-decay 1e-2]
                            #:foreach? [foreach? #t]
                            #:fused? [fused? #f])
  (->i ([params (fused?) (fused-parameters/c fused?)])
       (#:lr [lr (>=/c 0)] #:beta1 [beta1 real?] #:beta2 [beta2 real?]
        #:eps [eps real?] #:weight-decay [weight-decay (>=/c 0)]
        #:foreach? [foreach? boolean?] #:fused? [fused? boolean?])
       #:pre/name (foreach? fused?)
       "fused and foreach cannot both be requested"
       (not (and (eq? foreach? #t) (eq? fused? #t)))
       [result adamw?])
  (error 'unimplemented))

(define (adam-do-step! opt)
  (with-no-grad
    (define b1 (adam-beta1 opt))
    (define b2 (adam-beta2 opt))
    (define lr (adam-lr opt))
    (define eps (adam-eps opt))
    (follow-moves! opt)
    (for ([p (in-list (adam-params opt))])
      (with-gradient (g p)
        (define t (add1 (hash-ref (adam-steps opt) p 0)))
        (hash-set! (adam-steps opt) p t)
        (define bc1 (- 1.0 (expt b1 t)))
        (define bc2 (- 1.0 (expt b2 t)))
        (define d (decayed g p (adam-weight-decay opt)))
        (define m (hash-ref! (adam-m opt) p (lambda () (zeros-like p))))
        (define v (hash-ref! (adam-v opt) p (lambda () (zeros-like p))))
        (mul! m b1)
        (sub! m d (- b1 1.0))
        (mul! v b2)
        (addcmul! v d d (- 1.0 b2))
        (define denom (sqrt v))
        (mul! denom (/ 1.0 (expt bc2 0.5)))
        (sub! denom (adam-one opt) (- eps))
        (addcdiv! p m denom (- (/ lr bc1)))))))

(struct rmsprop tracked (params [lr #:mutable] alpha eps weight-decay
                                momentum foreach? sq bufs one)
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
                              #:momentum [momentum 0.0]
                              #:foreach? [foreach? #t])
  (->* [(listof tensor?)]
       [#:lr (>=/c 0) #:alpha (>=/c 0) #:eps (>=/c 0)
        #:weight-decay (>=/c 0)
        #:momentum (>=/c 0) #:foreach? boolean?]
       rmsprop?)
  (define sq (make-hasheq))
  (define bufs (make-hasheq))
  (make-rmsprop (list sq bufs) #f params lr alpha eps weight-decay momentum
                foreach? sq bufs (unit-scalar)))

(define (rmsprop-do-step! opt)
  (with-no-grad
    (define alpha (rmsprop-alpha opt))
    (define mu (rmsprop-momentum opt))
    (define lr (rmsprop-lr opt))
    (follow-moves! opt)
    (for ([p (in-list (rmsprop-params opt))])
      (with-gradient (g p)
        (define d (decayed g p (rmsprop-weight-decay opt)))
        (define v (hash-ref! (rmsprop-sq opt) p (lambda () (zeros-like p))))
        (mul! v alpha)
        (addcmul! v d d (- 1.0 alpha))
        (define avg (sqrt v))
        (sub! avg (rmsprop-one opt) (- (rmsprop-eps opt)))
        (cond
          [(zero? mu) (addcdiv! p d avg (- lr))]
          [else
           (define buf
             (hash-ref! (rmsprop-bufs opt) p (lambda () (zeros-like p))))
           (mul! buf mu)
           (addcdiv! buf d avg 1.0)
           (sub! p buf lr)])))))

(define/contract-out (step! opt) ;; noqa
  (-> optimizer? void?)
  (optimizer-step! opt)
  (when (or (sgd? opt) (adam? opt) (rmsprop? opt))
    (log-optimizer-step! opt (optimizer-parameters opt))))

(define/contract-out (zero-grads! opt #:set-to-none? [set-to-none? #t]) ;; noqa
  (->* [optimizer?] [#:set-to-none? boolean?] void?)
  (define params (optimizer-parameters opt))
  (if set-to-none?
      (reset-grads! params)
      (for-each zero-grad! params)))

(define/contract-out (learning-rate opt) ;; noqa
  (-> optimizer? real?)
  (optimizer-lr opt))

(define/contract-out (set-learning-rate! opt lr) ;; noqa
  (-> optimizer? (>=/c 0) void?)
  (optimizer-set-lr! opt lr))
