#lang racket/base

(require (only-in racket/contract/base -> ->* ->i >=/c contract-name non-empty-listof or/c)
         (only-in racket/contract/combinator contract-first-order-passes?)
         (only-in racket/list range)
         (only-in "../nn/optim.rkt"
                  has-hyperparameter? hyperparameter-value/c param-group-count param-group-set!)
         (only-in "optimization.rkt" set-hparam!)
         (only-in "private/types.rkt" callback? define-named-contract trainer-hparam-names)
         (only-in "run.rkt" interval/c run-epoch run-global-step run-optimizers run-trainer)
         (only-in "trainer.rkt" define-callback define-hook define-init run))

(define-named-contract schedulable-key/c
  (or/c 'lr 'weight-decay 'momentum 'beta1 'beta2 'eps 'alpha))

(define real-curve/c (-> exact-nonnegative-integer? real?))

(define nonnegative-curve/c (-> exact-nonnegative-integer? (>=/c 0)))

(define (curve/c key)
  (if (memq key '(beta1 beta2 eps)) real-curve/c nonnegative-curve/c))

(define-callback hparam-scheduler (name curve interval)
  #:contract (->* (symbol? (-> exact-nonnegative-integer? real?)) (#:interval interval/c)
                  callback?)
  #:kind plain
  (define-init (n c #:interval [i 'step])
    (set! name n)
    (set! curve c)
    (set! interval i))
  (define-hook (on-train-start)
    (define tr (run-trainer run))
    (unless (memq name (trainer-hparam-names tr))
      (raise-arguments-error 'hparam-scheduler
                             "the trainer has no define-hparams field of this name"
                             "name" name "hparams" (trainer-hparam-names tr))))
  (define-hook (on-train-epoch-start)
    (when (eq? interval 'epoch)
      (write-hparam! run name curve (run-epoch run))))
  (define-hook (on-train-batch-start . _)
    (when (eq? interval 'step)
      (write-hparam! run name curve (run-global-step run)))))

(define (write-hparam! r name curve t)
  (set-hparam! (run-trainer r) name (curve t)))

(define (optimizer-at r index)
  (define optimizers (run-optimizers r))
  (and (< index (length optimizers)) (list-ref optimizers index)))

(define (missing-group opt groups)
  (and groups (findf (lambda (g) (>= g (param-group-count opt))) groups)))

(define (check-schedulable! r key index groups)
  (define opt (optimizer-at r index))
  (cond
    [(not opt)
     (raise-arguments-error 'param-group-scheduler
                            "define-optimizers returned no optimizer at this index"
                            "optimizer" index "optimizers" (length (run-optimizers r)))]
    [(missing-group opt groups)
     => (lambda (g)
          (raise-arguments-error 'param-group-scheduler
                                 "the optimizer has no parameter group at this index"
                                 "group" g "groups" (param-group-count opt)))]
    [(not (has-hyperparameter? opt key))
     (raise-arguments-error 'param-group-scheduler "the optimizer has no such hyperparameter"
                            "key" key)]
    [else (void)]))

(define (write-groups! r key curve index groups t)
  (define opt (optimizer-at r index))
  (define value/c (hyperparameter-value/c opt key))
  (define v (curve t))
  (unless (contract-first-order-passes? value/c v)
    (raise-arguments-error 'param-group-scheduler
                           "the curve's value is out of range for this optimizer's hyperparameter"
                           "key" key "at" t "value" v "expected" (contract-name value/c)))
  (for ([g (in-list (or groups (range (param-group-count opt))))])
    (param-group-set! opt g key v)))

(define-callback param-group-scheduler (key curve optimizer groups interval)
  #:contract (->i ([key schedulable-key/c] [curve (key) (curve/c key)])
                  (#:optimizer [optimizer exact-nonnegative-integer?]
                   #:groups [groups (or/c #f (non-empty-listof exact-nonnegative-integer?))]
                   #:interval [interval interval/c])
                  [_ callback?])
  #:kind plain
  (define-init (k c #:optimizer [o 0] #:groups [gs #f] #:interval [i 'step])
    (set! key k)
    (set! curve c)
    (set! optimizer o)
    (set! groups gs)
    (set! interval i))
  (define-hook (on-train-start)
    (check-schedulable! run key optimizer groups))
  (define-hook (on-train-epoch-start)
    (when (eq? interval 'epoch)
      (write-groups! run key curve optimizer groups (run-epoch run))))
  (define-hook (on-train-batch-start . _)
    (when (eq? interval 'step)
      (write-groups! run key curve optimizer groups (run-global-step run)))))
