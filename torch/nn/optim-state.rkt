#lang racket/base

(require (only-in racket/contract/base ->i contract? flat-named-contract)
         (only-in racket/match match)
         (only-in "../foreign.rkt" tensor?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "layer.rkt" layer? named-parameters)
         (only-in "optim.rkt" optimizer-parameters optimizer?))

(define slot-names '("momentum_buffer" "exp_avg" "exp_avg_sq" "square_avg" "step"))

(define (string-keyed? v)
  (and (hash? v) (for/and ([k (in-hash-keys v)]) (string? k))))

(define (slot-value? name v)
  (if (equal? name "step") (exact-nonnegative-integer? v) (tensor? v)))

(define (slots? v)
  (and (string-keyed? v)
       (for/and ([(name x) (in-hash v)])
         (and (member name slot-names) (slot-value? name x)))))

(define (parameter-states? v)
  (and (string-keyed? v) (for/and ([slots (in-hash-values v)]) (slots? slots))))

(define (betas? v)
  (match v
    [(list (? real?) (? real?)) #t]
    [_ #f]))

(define (group-value? key v)
  (match key
    ["params" (and (list? v) (andmap string? v))]
    ["nesterov" (boolean? v)]
    ["betas" (betas? v)]
    [(or "lr" "momentum" "weight_decay" "eps" "alpha" "initial_lr") (real? v)]
    [_ #f]))

(define (param-group-state? v)
  (and (string-keyed? v)
       (hash-has-key? v "params")
       (for/and ([(key x) (in-hash v)]) (group-value? key x))))

(define (optimizer-state-shape? v)
  (and (hash? v) (immutable? v) (hash-equal? v)
       (= 2 (hash-count v))
       (parameter-states? (hash-ref v "state" #f))
       (match (hash-ref v "param_groups" #f)
         [(list (? param-group-state?) ..1) #t]
         [_ #f])))

(define (parameters-named? opt model)
  (define named
    (for/hasheq ([entry (in-list (named-parameters model))])
      (values (cdr entry) #t)))
  (for/and ([p (in-list (optimizer-parameters opt))])
    (hash-ref named p #f)))

(define/contract-out optimizer-state/c ;; noqa
  contract?
  (flat-named-contract 'optimizer-state/c optimizer-state-shape?))

(define/contract-out (optimizer-state-dict opt model) ;; noqa
  (->i ([opt optimizer?] [model layer?])
       #:pre/name (opt model) "every parameter of the optimizer is a named parameter of the model"
       (parameters-named? opt model)
       [_ optimizer-state/c])
  (error 'unimplemented))

(define/contract-out (load-optimizer-state! opt model state) ;; noqa
  (->i ([opt optimizer?] [model layer?] [state optimizer-state/c])
       #:pre/name (opt model) "every parameter of the optimizer is a named parameter of the model"
       (parameters-named? opt model)
       [_ void?])
  (error 'unimplemented))
