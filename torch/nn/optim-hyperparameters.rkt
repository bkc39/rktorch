#lang racket/base

(require (only-in racket/contract/base -> ->i >=/c listof or/c unsupplied-arg?)
         (only-in racket/match match)
         (only-in "../private/contract.rkt" define/contract-out supplied-or)
         (only-in "optim-groups.rkt" store-groups store-ref store-set! store-set-all!)
         (only-in "optim.rkt"
                  group-index? has-hyperparameter? hyperparameter-value/c lr-optimizer/c
                  optimizer-store optimizer?)
         (only-in "param-group.rkt" hyperparameter-key/c param-group?))

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
  (->i ([opt lr-optimizer/c]) (#:group [group exact-nonnegative-integer?])
       #:pre/name (opt group) "the optimizer has no parameter group with this index"
       (or (unsupplied-arg? group) (group-index? opt group))
       [_ real?])
  (store-ref (optimizer-store opt) group 'lr))

(define/contract-out (set-learning-rate! opt lr #:group [group #f]) ;; noqa
  (->i ([opt lr-optimizer/c] [lr (>=/c 0)])
       (#:group [group (or/c #f exact-nonnegative-integer?)])
       #:pre/name (opt group) "the optimizer has no parameter group with this index"
       (match (supplied-or group)
         [#f #t]
         [i (group-index? opt i)])
       [_ void?])
  (if group
      (store-set! (optimizer-store opt) group 'lr lr)
      (store-set-all! (optimizer-store opt) 'lr lr)))
