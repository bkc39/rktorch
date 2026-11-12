#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i >=/c non-empty-listof or/c)
         (only-in racket/list range)
         (only-in "../nn/optim.rkt" has-hyperparameter? param-group-count param-group-set!)
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

(define (scheduled-at r interval at)
  (and (eq? interval at)
       (if (eq? at 'step) (run-global-step r) (run-epoch r))))

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
    (write-hparam! run name curve (scheduled-at run interval 'epoch)))
  (define-hook (on-train-batch-start . _)
    (write-hparam! run name curve (scheduled-at run interval 'step))))

(define (write-hparam! r name curve at)
  (when at
    (set-hparam! (run-trainer r) name (curve at))))

(define (selected-groups opt groups)
  (or groups (range (param-group-count opt))))

(define (check-schedulable! r key index groups)
  (define optimizers (run-optimizers r))
  (define opt (and (< index (length optimizers)) (list-ref optimizers index)))
  (define missing
    (and opt (findf (lambda (g) (>= g (param-group-count opt))) (or groups '()))))
  (define problem
    (cond
      [(not opt)
       (list "define-optimizers returned no optimizer at this index"
             "optimizer" index "optimizers" (length optimizers))]
      [missing
       (list "the optimizer has no parameter group at this index"
             "group" missing "groups" (param-group-count opt))]
      [(not (has-hyperparameter? opt key))
       (list "the optimizer has no such hyperparameter" "key" key)]
      [else #f]))
  (when problem
    (apply raise-arguments-error 'param-group-scheduler problem)))

(define (write-groups! r key curve index groups at)
  (when at
    (define opt (list-ref (run-optimizers r) index))
    (define v (curve at))
    (for ([g (in-list (selected-groups opt groups))])
      (param-group-set! opt g key v))))

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
    (write-groups! run key curve optimizer groups (scheduled-at run interval 'epoch)))
  (define-hook (on-train-batch-start . _)
    (write-groups! run key curve optimizer groups (scheduled-at run interval 'step))))
