#lang racket/base

(require (only-in racket/contract/base
                  ->i contract? list/c listof non-empty-listof rename-contract)
         (only-in "../foreign.rkt" tensor?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "../private/not-implemented.rkt" not-implemented)
         (only-in "layer.rkt" layer? named-parameters)
         (only-in "optim.rkt" optimizer-parameters optimizer?)
         (only-in "private/state-shape.rkt" record/c state-real/c string-keyed/c))

(define slots/c
  (record/c (hash "momentum_buffer" tensor? "exp_avg" tensor? "exp_avg_sq" tensor?
                  "square_avg" tensor? "step" exact-nonnegative-integer?)
            #:required '()))

(define param-group/c
  (record/c (hash "params" (listof string?) "nesterov" boolean?
                  "betas" (list/c state-real/c state-real/c)
                  "lr" state-real/c "momentum" state-real/c "weight_decay" state-real/c
                  "eps" state-real/c "alpha" state-real/c "initial_lr" state-real/c)
            #:required '("params")))

(define (parameters-named? opt model)
  (define named
    (for/hasheq ([entry (in-list (named-parameters model))])
      (values (cdr entry) #t)))
  (for/and ([p (in-list (optimizer-parameters opt))])
    (hash-ref named p #f)))

(define/contract-out optimizer-state/c ;; noqa
  contract?
  (rename-contract (record/c (hash "state" (string-keyed/c slots/c)
                                  "param_groups" (non-empty-listof param-group/c)))
                   'optimizer-state/c))

(define/contract-out (optimizer-state-dict opt model) ;; noqa
  (->i ([opt optimizer?] [model layer?])
       #:pre/name (opt model) "every parameter of the optimizer is a named parameter of the model"
       (parameters-named? opt model)
       [_ optimizer-state/c])
  (not-implemented 'optimizer-state-dict 'T3))

(define/contract-out (load-optimizer-state! opt model state) ;; noqa
  (->i ([opt optimizer?] [model layer?] [state optimizer-state/c])
       #:pre/name (opt model) "every parameter of the optimizer is a named parameter of the model"
       (parameters-named? opt model)
       [_ void?])
  (not-implemented 'load-optimizer-state! 'T3))
