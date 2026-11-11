#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i >=/c and/c any/c contract? contract-out flat-named-contract
                  hash/c list/c listof non-empty-listof or/c property/c rename-contract)
         (only-in racket/list append-map check-duplicates remove-duplicates)
         (only-in "../foreign.rkt" dtype tensor?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "layer.rkt" layer?))

(provide (contract-out [param-group? (-> any/c boolean?)]))

(module+ private
  (provide floating-params/c
           grouped?
           groups-distinct?
           groups-keys-within?
           groups-setting-holds?
           param-group-settings
           params/c))

(struct param-group (tensors settings)
  #:constructor-name make-param-group ;; noqa
  #:name param-group-struct) ;; noqa

(define/contract-out hyperparameter-key/c
  contract?
  (flat-named-contract 'hyperparameter-key/c
                       (or/c 'lr 'weight-decay 'momentum 'nesterov? 'beta1 'beta2 'eps
                             'alpha)))

(define/contract-out (param-group params #:lr [lr #f] #:weight-decay [weight-decay #f]
                                  #:momentum [momentum #f] #:nesterov? [nesterov? #f]
                                  #:beta1 [beta1 #f] #:beta2 [beta2 #f]
                                  #:eps [eps #f] #:alpha [alpha #f])
  (->i ([params (listof tensor?)])
       (#:lr [lr (>=/c 0)] #:weight-decay [weight-decay (>=/c 0)]
        #:momentum [momentum (>=/c 0)] #:nesterov? [nesterov? boolean?]
        #:beta1 [beta1 real?] #:beta2 [beta2 real?] #:eps [eps real?]
        #:alpha [alpha (>=/c 0)])
       [result param-group?])
  (error 'unimplemented))

(define/contract-out (param-group-params g)
  (-> param-group? (listof tensor?))
  (error 'unimplemented))

(define/contract-out (param-group-hyperparameters g)
  (-> param-group? (hash/c symbol? (or/c real? boolean?) #:immutable #t))
  (error 'unimplemented))

(define/contract-out (weight-decay-param-groups model #:weight-decay [weight-decay #f]
                                                #:skip [skip '("cls-token" "pos-embed")])
  (->* (layer?) (#:weight-decay (or/c #f (>=/c 0)) #:skip (listof string?))
       (list/c param-group? param-group?))
  (error 'unimplemented))

(define floating-tensor/c
  (rename-contract
   (and/c tensor? (property/c dtype (or/c 'float16 'bfloat16 'float32 'float64)))
   'floating-tensor/c))

(define floating-tensors/c (listof floating-tensor/c))

(define floating-group/c
  (rename-contract (and/c param-group? (property/c param-group-tensors floating-tensors/c))
                   'floating-param-group/c))

(define params/c (or/c (listof tensor?) (non-empty-listof param-group?)))

(define floating-params/c (or/c floating-tensors/c (non-empty-listof floating-group/c)))

(define (grouped? params)
  (ormap param-group? params))

(define (groups-of params)
  (filter param-group? params))

(define (groups-distinct? params)
  (not (check-duplicates (append-map (lambda (g) (remove-duplicates (param-group-tensors g) eq?))
                                     (groups-of params))
                         eq?)))

(define (groups-keys-within? params keys)
  (for*/and ([g (in-list (groups-of params))]
             [k (in-hash-keys (param-group-settings g))])
    (and (memq k keys) #t)))

(define (groups-setting-holds? params key holds?)
  (for/and ([g (in-list (groups-of params))])
    (define v (hash-ref (param-group-settings g) key #f))
    (or (not v) (holds? v))))
