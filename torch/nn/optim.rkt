#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i >=/c any/c contract-out flat-named-contract listof
                  unsupplied-arg?)
         (only-in racket/generic define-generics)
         (only-in racket/flonum flexpt)
         (only-in racket/list group-by)
         (only-in "../foreign.rkt"
                  addcdiv! addcmul! copy! cpu-device device dtype full mul!
                  sqrt sub! tensor? to with-no-grad zero-grad!)
         (only-in "../foreign/autograd-ops.rkt"
                  reset-grads! with-gradient with-gradients)
         (only-in "../foreign/ops.rkt" move-generation)
         (prefix-in g: (only-in "../generated.rkt"
                                add-tensor add-tensor! div-scalar! lerp-scalar!))
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "../private/step-log.rkt" log-optimizer-step!)
         (only-in "optim-foreach.rkt"
                  adam-foreach! adam-fused! bias-correction rmsprop-foreach!
                  sgd-foreach! sgd-fused! slot! slot-ref step-count!))

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

(struct tracked (state keys [generation #:mutable]))

(define (optimizer-state opt)
  (for*/list ([table (in-hash-values (tracked-state opt))]
              [s (in-hash-values table)]
              #:when (tensor? s))
    s))

(define (placed-as name s p)
  (cond
    [(not (tensor? s)) s]
    [(eq? name 'step) (placed-on s (device p) (dtype s))]
    [else (placed-on s (device p) (dtype p))]))

(define (placed-on s dev dt)
  (if (and (equal? (device s) dev) (eq? (dtype s) dt)) s (to s dev dt)))

(define (follow-moves! opt)
  (define now (move-generation))
  (cond
    [(eqv? (tracked-generation opt) now) (void)]
    [else
     (for* ([(name table) (in-hash (tracked-state opt))]
            [p (in-list (hash-keys table))])
       (hash-update! table p (lambda (s) (placed-as name s p))))
     (hash-clear! (tracked-keys opt))
     (set-tracked-generation! opt now)]))

(define (bucketed opt qs gs)
  (define keys (tracked-keys opt))
  (define (key-of qg)
    (hash-ref! keys (car qg) (lambda () (cons (device (car qg)) (dtype (car qg))))))
  (for/list ([bucket (in-list (group-by key-of (map cons qs gs)))])
    (cons (map car bucket) (map cdr bucket))))

(define (multi-body foreach? fused? foreach-body fused-body)
  (cond
    [fused? fused-body]
    [foreach? foreach-body]
    [else #f]))

(define (step-by! opt params one! many!)
  (with-no-grad
    (follow-moves! opt)
    (cond
      [many!
       (with-gradients ([qs gs] params)
         (for ([bucket (in-list (bucketed opt qs gs))])
           (many! (car bucket) (cdr bucket))))]
      [else
       (for ([p (in-list params)])
         (with-gradient (g p)
           (one! p g)))])))

(define (unit-scalar)
  (full 1.0 #:device (cpu-device) #:dtype 'float32))

(define (decayed g p weight-decay)
  (if (zero? weight-decay) g (g:add-tensor g p weight-decay)))

(define floating-tensor/c
  (flat-named-contract 'floating-tensor/c
                       (lambda (v) (and (tensor? v)
                                        (memq (dtype v) '(float16 bfloat16 float32 float64))
                                        #t))))

(define (fused-parameters/c fused?)
  (if (eq? fused? #t) (listof floating-tensor/c) (listof tensor?)))

(struct sgd tracked (params [lr #:mutable] momentum nesterov? weight-decay
                            foreach? fused?)
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
  (make-sgd (make-hasheq) (make-hasheq) #f params lr momentum nesterov?
            weight-decay foreach? fused?))

(define (momentum-buffer! state p d momentum)
  (cond
    [(slot-ref state 'momentum-buffer p)
     => (lambda (buf)
          (mul! buf momentum)
          (g:add-tensor! buf d 1.0)
          buf)]
    [else
     (define fresh (slot! state 'momentum-buffer p))
     (copy! fresh d)
     fresh]))

(define (sgd-loop! state p g #:lr lr #:momentum momentum #:nesterov? nesterov?
                   #:weight-decay weight-decay)
  (define d (decayed g p weight-decay))
  (define step
    (cond
      [(zero? momentum) d]
      [nesterov? (g:add-tensor d (momentum-buffer! state p d momentum) momentum)]
      [else (momentum-buffer! state p d momentum)]))
  (g:add-tensor! p step (- lr)))

(define (sgd-do-step! opt)
  (define state (tracked-state opt))
  (define lr (real->double-flonum (sgd-lr opt)))
  (define momentum (real->double-flonum (sgd-momentum opt)))
  (define weight-decay (real->double-flonum (sgd-weight-decay opt)))
  (define nesterov? (sgd-nesterov? opt))
  (define many (multi-body (sgd-foreach? opt) (sgd-fused? opt)
                           sgd-foreach! sgd-fused!))
  (step-by! opt (sgd-params opt)
            (lambda (p g)
              (sgd-loop! state p g #:lr lr #:momentum momentum
                         #:nesterov? nesterov? #:weight-decay weight-decay))
            (and many
                 (lambda (ps gs)
                   (many state ps gs #:lr lr #:momentum momentum
                         #:nesterov? nesterov? #:weight-decay weight-decay)))))

(struct adam tracked (params [lr #:mutable] beta1 beta2 eps weight-decay
                             foreach? fused? one)
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
  (make-adam (make-hasheq) (make-hasheq) #f params lr beta1 beta2 eps
             weight-decay foreach? fused? (unit-scalar)))

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
  (make-adamw (make-hasheq) (make-hasheq) #f params lr beta1 beta2 eps
              weight-decay foreach? fused? (unit-scalar)))

(define (adam-loop! state p g one #:lr lr #:beta1 beta1 #:beta2 beta2 #:eps eps
                    #:weight-decay weight-decay #:decoupled? decoupled?)
  (define t (step-count! state p))
  (define d
    (cond
      [decoupled?
       (mul! p (- 1.0 (* lr weight-decay)))
       g]
      [else (decayed g p weight-decay)]))
  (define m (slot! state 'exp-avg p))
  (define v (slot! state 'exp-avg-sq p))
  (g:lerp-scalar! m d (- 1.0 beta1))
  (mul! v beta2)
  (addcmul! v d d (- 1.0 beta2))
  (define denom (sqrt v))
  (g:div-scalar! denom (flexpt (bias-correction beta2 t) 0.5))
  (sub! denom one (- eps))
  (addcdiv! p m denom (* (/ lr (bias-correction beta1 t)) -1.0)))

(define (adam-do-step! opt)
  (define state (tracked-state opt))
  (define lr (real->double-flonum (adam-lr opt)))
  (define beta1 (real->double-flonum (adam-beta1 opt)))
  (define beta2 (real->double-flonum (adam-beta2 opt)))
  (define eps (real->double-flonum (adam-eps opt)))
  (define weight-decay (real->double-flonum (adam-weight-decay opt)))
  (define decoupled? (adamw? opt))
  (define many (multi-body (adam-foreach? opt) (adam-fused? opt)
                           adam-foreach! adam-fused!))
  (step-by! opt (adam-params opt)
            (lambda (p g)
              (adam-loop! state p g (adam-one opt) #:lr lr #:beta1 beta1
                          #:beta2 beta2 #:eps eps #:weight-decay weight-decay
                          #:decoupled? decoupled?))
            (and many
                 (lambda (ps gs)
                   (many state ps gs #:lr lr #:beta1 beta1 #:beta2 beta2
                         #:eps eps #:weight-decay weight-decay
                         #:decoupled? decoupled?)))))

(struct rmsprop tracked (params [lr #:mutable] alpha eps weight-decay
                                momentum foreach? one)
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
  (make-rmsprop (make-hasheq) (make-hasheq) #f params lr alpha eps
                weight-decay momentum foreach? (unit-scalar)))

(define (rmsprop-loop! state p g one #:lr lr #:alpha alpha #:eps eps
                       #:weight-decay weight-decay #:momentum momentum)
  (define d (decayed g p weight-decay))
  (define v (slot! state 'square-avg p))
  (mul! v alpha)
  (addcmul! v d d (- 1.0 alpha))
  (define avg (sqrt v))
  (sub! avg one (- eps))
  (cond
    [(zero? momentum) (addcdiv! p d avg (- lr))]
    [else
     (define buf (slot! state 'momentum-buffer p))
     (mul! buf momentum)
     (addcdiv! buf d avg 1.0)
     (g:add-tensor! p buf (- lr))]))

(define (rmsprop-do-step! opt)
  (define state (tracked-state opt))
  (define lr (real->double-flonum (rmsprop-lr opt)))
  (define alpha (real->double-flonum (rmsprop-alpha opt)))
  (define eps (real->double-flonum (rmsprop-eps opt)))
  (define weight-decay (real->double-flonum (rmsprop-weight-decay opt)))
  (define momentum (real->double-flonum (rmsprop-momentum opt)))
  (step-by! opt (rmsprop-params opt)
            (lambda (p g)
              (rmsprop-loop! state p g (rmsprop-one opt) #:lr lr #:alpha alpha
                             #:eps eps #:weight-decay weight-decay
                             #:momentum momentum))
            (and (rmsprop-foreach? opt)
                 (lambda (ps gs)
                   (rmsprop-foreach! state ps gs #:lr lr #:alpha alpha #:eps eps
                                     #:weight-decay weight-decay
                                     #:momentum momentum)))))

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
