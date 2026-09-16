#lang racket/base

(require (only-in racket/contract/base -> ->* any/c contract-out listof)
         (only-in racket/generic define-generics)
         (only-in "../foreign.rkt"
                  addcdiv! addcmul! full maybe-grad mul! sqrt sub!
                  tensor-device tensor-dtype tensor? to with-no-grad
                  zero-grad! zeros-like)
         (only-in "../private/contract.rkt" define/contract-out))

(provide gen:optimizer
         optimizer?
         sgd-lr
         (contract-out [sgd? (-> any/c boolean?)]
                       [adam? (-> any/c boolean?)]))

(define-generics optimizer
  (optimizer-step! optimizer)
  (optimizer-parameters optimizer))

(struct sgd (params lr)
  #:constructor-name make-sgd
  #:name sgd-optimizer ;; noqa
  #:methods gen:optimizer
  [(define (optimizer-parameters opt) (sgd-params opt))
   (define (optimizer-step! opt)
     (with-no-grad
       (for ([p (in-list (sgd-params opt))])
         (define g (maybe-grad p))
         (when g
           (sub! p g (sgd-lr opt))))))])

(define/contract-out (sgd params #:lr lr) ;; noqa
  (-> (listof tensor?) #:lr real? sgd?)
  (make-sgd params lr))

(struct adam (params lr beta1 beta2 eps step-box m v scalars)
  #:constructor-name make-adam
  #:name adam-optimizer ;; noqa
  #:methods gen:optimizer
  [(define (optimizer-parameters opt) (adam-params opt))
   (define (optimizer-step! opt) (adam-do-step! opt))])

(define/contract-out (adam params ;; noqa
                           #:lr [lr 1e-3]
                           #:beta1 [beta1 0.9]
                           #:beta2 [beta2 0.999]
                           #:eps [eps 1e-8])
  (->* [(listof tensor?)]
       [#:lr real? #:beta1 real? #:beta2 real? #:eps real?]
       adam?)
  (make-adam params lr beta1 beta2 eps (box 0) (make-hasheq) (make-hasheq)
             (make-hash)))

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
  (hash-ref! table (list value dev dt)
             (lambda () (full value #:device dev #:dtype dt))))

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
      (define g (maybe-grad p))
      (when g
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

(define/contract-out (step! opt) ;; noqa
  (-> optimizer? void?)
  (optimizer-step! opt))

(define/contract-out (zero-grads! opt) ;; noqa
  (-> optimizer? void?)
  (for-each zero-grad! (optimizer-parameters opt)))
