#lang racket/base

(require (for-syntax racket/base
                     ;; whole-module require on purpose
                     syntax/parse/pre)
         (only-in racket/contract/base
                  -> ->* ->i >=/c any/c contract-out listof or/c unsupplied-arg?)
         (only-in racket/flonum flexpt)
         (only-in racket/generic define-generics)
         (only-in racket/list group-by)
         (only-in racket/match match match-define)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../foreign.rkt"
                  addcdiv! addcmul! copy! cpu-device device dtype full mul!
                  sqrt sub! tensor? to with-no-grad zero-grad!)
         (only-in "../foreign/autograd-ops.rkt"
                  reset-grads! with-gradient with-gradients)
         (only-in "../foreign/ops.rkt" move-generation)
         (prefix-in g: (only-in "../generated.rkt"
                                add-tensor add-tensor! div-scalar! lerp-scalar!))
         (only-in "../private/contract.rkt" define/checked-out define/contract-out supplied-or)
         (only-in "../private/step-log.rkt" log-optimizer-step!)
         (only-in "optim-groups.rkt"
                  group-store store-count store-groups store-params store-partition store-ref
                  store-set! store-set-all!)
         (only-in "param-group.rkt" hyperparameter-key/c param-group?)
         (only-in (submod "param-group.rkt" private)
                  floating-params/c groups-distinct? groups-keys-within? groups-setting-holds?
                  param-group-settings param-group-tensors params/c)
         (only-in "optim-foreach.rkt"
                  adam-foreach! adam-fused! bias-correction rmsprop-foreach!
                  sgd-foreach! sgd-fused! slot! slot-ref step-count!))

(provide gen:optimizer
         optimizer?
         optimizer-step!
         optimizer-parameters
         optimizer-store
         optimizer-root
         group-index?
         has-hyperparameter?
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
  (optimizer-store optimizer)
  (optimizer-root optimizer)
  #:fallbacks [(define (optimizer-parameters opt) (store-params (tracked-store opt)))
               (define (optimizer-store opt) (tracked-store opt))
               (define (optimizer-root opt) opt)])

(struct tracked (state keys [generation #:mutable] store))

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

(define (placement q)
  (cons (device q) (dtype q)))

(define (update-buckets! opt qs gs update!)
  (define keys (tracked-keys opt))
  (define (key-of qg)
    (match-define (cons q _) qg)
    (hash-ref! keys q (lambda () (placement q))))
  (for ([bucket (in-list (group-by key-of (map cons qs gs)))])
    (define-values (bqs bgs)
      (for/lists (bqs bgs) ([qg (in-list bucket)])
        (match-define (cons q g) qg)
        (values q g)))
    (update! bqs bgs)))

(define (update-mode foreach? fused?)
  (cond
    [fused? 'fused]
    [foreach? 'foreach]
    [else 'loop]))

(define (step-by! opt mode #:loop loop! #:foreach foreach! #:fused [fused! #f])
  (define st (tracked-store opt))
  (define (by-group! update!)
    (with-gradients ([qs gs] (store-params st))
      (for ([part (in-list (store-partition st qs gs))])
        (match-define (list h gqs ggs) part)
        (update-buckets! opt gqs ggs (lambda (bqs bgs) (update! h bqs bgs))))))
  (with-no-grad
    (follow-moves! opt)
    (match mode
      ['loop
       (for* ([group (in-list (store-groups st))]
              [h (in-value (param-group-settings group))]
              [p (in-list (param-group-tensors group))])
         (with-gradient (g p)
           (loop! h p g)))]
      ['foreach (by-group! foreach!)]
      ['fused (by-group! fused!)])))

(define (unit-scalar)
  (full 1.0 #:device (cpu-device) #:dtype 'float32))

(define (decayed g p weight-decay)
  (if (zero? weight-decay) g (g:add-tensor g p weight-decay)))

(define (explicit? v)
  (and (not (unsupplied-arg? v)) v))

(define (fused-parameters/c fused?)
  (if (explicit? fused?) floating-params/c params/c))

(define sgd-keys '(lr momentum nesterov? weight-decay))
(define adam-keys '(lr beta1 beta2 eps weight-decay))
(define rmsprop-keys '(lr alpha eps weight-decay momentum))

(define (hyperparameter-keys opt)
  (define root (optimizer-root opt))
  (cond
    [(sgd? root) sgd-keys]
    [(adam? root) adam-keys]
    [(rmsprop? root) rmsprop-keys]
    [else '()]))

(define (has-hyperparameter? opt key)
  (and (memq key (hyperparameter-keys opt)) #t))

(define (group-index? opt i)
  (< i (param-group-count opt)))

(define nonnegative/c (>=/c 0))

(define (hyperparameter-value/c opt key)
  (match key
    ['nesterov? boolean?]
    [(or 'beta1 'beta2) real?]
    ['eps (if (rmsprop? (optimizer-root opt)) nonnegative/c real?)]
    [_ nonnegative/c]))

(begin-for-syntax
  (define-syntax-class hyperparameter
    #:attributes ([formal 1] [required 1] [optional 1])
    (pattern [kw:keyword name:id ctc:expr]
             #:with (formal ...) #'(kw name)
             #:with (required ...) #'(kw [name ctc])
             #:with (optional ...) #'())
    (pattern [kw:keyword name:id ctc:expr default:expr]
             #:with (formal ...) #'(kw [name default])
             #:with (required ...) #'()
             #:with (optional ...) #'(kw [name ctc]))))

(define-syntax-parse-rule
  (define-fusable-optimizer (name:id params:id h:hyperparameter ...)
    (~seq #:pre/name pre-ids:expr pre-message:expr pre-test:expr) ...
    #:keys keys:expr
    #:mode mode:id
    #:result result?:expr
    body:expr ...+)
  (define/contract-out (name params (~@ h.formal ...) ...
                             #:foreach? [foreach? #t] #:fused? [fused? #f])
    (->i ([params (fused?) (fused-parameters/c fused?)] (~@ h.required ...) ...)
         ((~@ h.optional ...) ...
          #:foreach? [foreach? boolean?] #:fused? [fused? boolean?])
         (~@ #:pre/name pre-ids pre-message pre-test) ...
         #:pre/name (params) "a parameter appears in more than one parameter group"
         (groups-distinct? params)
         #:pre/name (params)
         "a parameter group sets a hyperparameter this optimizer does not have"
         (groups-keys-within? params keys)
         #:pre/name (foreach? fused?)
         "fused and foreach cannot both be requested"
         (not (and (explicit? foreach?) (explicit? fused?)))
         [result result?])
    (define mode (update-mode foreach? fused?))
    body ...))

(struct sgd tracked (mode)
  #:constructor-name make-sgd
  #:name sgd-optimizer
  #:methods gen:optimizer
  [(define (optimizer-step! opt) (sgd-do-step! opt))])

(define-fusable-optimizer (sgd params
                               [#:lr lr (>=/c 0)]
                               [#:momentum momentum (>=/c 0) 0.0]
                               [#:nesterov? nesterov? boolean? #f]
                               [#:weight-decay weight-decay (>=/c 0) 0.0])
  #:pre/name (momentum nesterov?)
  "Nesterov momentum requires a momentum"
  (or (not (explicit? nesterov?))
      (and (not (unsupplied-arg? momentum)) (> momentum 0)))
  #:keys sgd-keys
  #:mode mode
  #:result sgd?
  (make-sgd (make-hasheq) (make-hasheq) #f
            (group-store params (hasheq 'lr lr 'momentum momentum 'nesterov? nesterov?
                                        'weight-decay weight-decay))
            mode))

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
  (define ((with-hyperparameters update!) h ps gs)
    (update! (tracked-state opt) ps gs #:lr (hash-ref h 'lr)
             #:momentum (hash-ref h 'momentum) #:nesterov? (hash-ref h 'nesterov?)
             #:weight-decay (hash-ref h 'weight-decay)))
  (step-by! opt (sgd-mode opt)
            #:loop (with-hyperparameters sgd-loop!)
            #:foreach (with-hyperparameters sgd-foreach!)
            #:fused (with-hyperparameters sgd-fused!)))

(struct adam tracked (mode one)
  #:constructor-name make-adam
  #:name adam-optimizer
  #:methods gen:optimizer
  [(define (optimizer-step! opt) (adam-do-step! opt))])

(struct adamw adam-optimizer ()
  #:constructor-name make-adamw
  #:omit-define-syntaxes)

(define-syntax-parse-rule (define-adam name:id make:id result?:id
                            default-decay:expr)
  (define-fusable-optimizer (name params
                                  [#:lr lr (>=/c 0) 1e-3]
                                  [#:beta1 beta1 real? 0.9]
                                  [#:beta2 beta2 real? 0.999]
                                  [#:eps eps real? 1e-8]
                                  [#:weight-decay weight-decay (>=/c 0)
                                   default-decay])
    #:keys adam-keys
    #:mode mode
    #:result result?
    (make (make-hasheq) (make-hasheq) #f
          (group-store params (hasheq 'lr lr 'beta1 beta1 'beta2 beta2 'eps eps
                                      'weight-decay weight-decay))
          mode (unit-scalar))))

(define-adam adam make-adam adam? 0.0)
(define-adam adamw make-adamw adamw? 1e-2)

(define ((adam-loop! one) state p g #:lr lr #:beta1 beta1 #:beta2 beta2
                          #:eps eps #:weight-decay weight-decay
                          #:decoupled? decoupled?)
  (define t (step-count! state p))
  (define d
    (cond
      [(not decoupled?) (decayed g p weight-decay)]
      [(zero? weight-decay) g]
      [else
       (mul! p (- 1.0 (* lr weight-decay)))
       g]))
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
  (define ((with-hyperparameters update!) h ps gs)
    (update! (tracked-state opt) ps gs #:lr (hash-ref h 'lr)
             #:beta1 (hash-ref h 'beta1) #:beta2 (hash-ref h 'beta2)
             #:eps (hash-ref h 'eps) #:weight-decay (hash-ref h 'weight-decay)
             #:decoupled? (adamw? opt)))
  (step-by! opt (adam-mode opt)
            #:loop (with-hyperparameters (adam-loop! (adam-one opt)))
            #:foreach (with-hyperparameters adam-foreach!)
            #:fused (with-hyperparameters adam-fused!)))

(struct rmsprop tracked (mode one)
  #:constructor-name make-rmsprop
  #:name rmsprop-optimizer ;; noqa
  #:methods gen:optimizer
  [(define (optimizer-step! opt) (rmsprop-do-step! opt))])

(define/contract-out (rmsprop params ;; noqa
                              #:lr [lr 1e-2]
                              #:alpha [alpha 0.99]
                              #:eps [eps 1e-8]
                              #:weight-decay [weight-decay 0.0]
                              #:momentum [momentum 0.0]
                              #:foreach? [foreach? #t])
  (->i ([params params/c])
       (#:lr [lr (>=/c 0)] #:alpha [alpha (>=/c 0)] #:eps [eps (>=/c 0)]
        #:weight-decay [weight-decay (>=/c 0)]
        #:momentum [momentum (>=/c 0)] #:foreach? [foreach? boolean?])
       #:pre/name (params) "a parameter appears in more than one parameter group"
       (groups-distinct? params)
       #:pre/name (params)
       "a parameter group sets a hyperparameter this optimizer does not have"
       (groups-keys-within? params rmsprop-keys)
       #:pre/name (params) "a parameter group's eps is negative"
       (groups-setting-holds? params 'eps (lambda (e) (>= e 0)))
       [result rmsprop?])
  (make-rmsprop (make-hasheq) (make-hasheq) #f
                (group-store params (hasheq 'lr lr 'alpha alpha 'eps eps
                                            'weight-decay weight-decay 'momentum momentum))
                (update-mode foreach? #f) (unit-scalar)))

(define ((rmsprop-loop! one) state p g #:lr lr #:alpha alpha #:eps eps
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
  (define ((with-hyperparameters update!) h ps gs)
    (update! (tracked-state opt) ps gs #:lr (hash-ref h 'lr)
             #:alpha (hash-ref h 'alpha) #:eps (hash-ref h 'eps)
             #:weight-decay (hash-ref h 'weight-decay)
             #:momentum (hash-ref h 'momentum)))
  (step-by! opt (rmsprop-mode opt)
            #:loop (with-hyperparameters (rmsprop-loop! (rmsprop-one opt)))
            #:foreach (with-hyperparameters rmsprop-foreach!)))

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

(define/checked-out (param-group-count opt)
  (-> optimizer? exact-positive-integer?)
  (store-count (optimizer-store opt)))

(define/contract-out (optimizer-param-groups opt) ;; noqa
  (-> optimizer? (listof param-group?))
  (store-groups (optimizer-store opt)))

(define/contract-out (param-group-ref opt i key) ;; noqa
  (->i ([opt optimizer?] [i exact-nonnegative-integer?] [key hyperparameter-key/c])
       #:pre/name (opt i) "the optimizer has no parameter group with this index"
       (group-index? opt i)
       #:pre/name (opt key) "this optimizer has no such hyperparameter"
       (has-hyperparameter? opt key)
       [_ (or/c real? boolean?)])
  (store-ref (optimizer-store opt) i key))

(define/contract-out (param-group-set! opt i key v) ;; noqa
  (->i ([opt optimizer?] [i exact-nonnegative-integer?] [key hyperparameter-key/c]
        [v (opt key) (hyperparameter-value/c opt key)])
       #:pre/name (opt i) "the optimizer has no parameter group with this index"
       (group-index? opt i)
       #:pre/name (opt key) "this optimizer has no such hyperparameter"
       (has-hyperparameter? opt key)
       [_ void?])
  (store-set! (optimizer-store opt) i key v))

(define/contract-out (learning-rate opt #:group [group 0]) ;; noqa
  (->i ([opt optimizer?]) (#:group [group exact-nonnegative-integer?])
       #:pre/name (opt group) "the optimizer has no parameter group with this index"
       (or (unsupplied-arg? group) (group-index? opt group))
       [_ real?])
  (store-ref (optimizer-store opt) group 'lr))

(define/contract-out (set-learning-rate! opt lr #:group [group #f]) ;; noqa
  (->i ([opt optimizer?] [lr (>=/c 0)]) (#:group [group (or/c #f exact-nonnegative-integer?)])
       #:pre/name (opt group) "the optimizer has no parameter group with this index"
       (match (supplied-or group)
         [#f #t]
         [i (group-index? opt i)])
       [_ void?])
  (if group
      (store-set! (optimizer-store opt) group 'lr lr)
      (store-set-all! (optimizer-store opt) 'lr lr)))
