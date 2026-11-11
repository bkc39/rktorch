#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i >=/c and/c any/c contract? contract-out flat-named-contract
                  hash/c list/c listof non-empty-listof or/c property/c rename-contract)
         (only-in racket/list append-map check-duplicates partition remove-duplicates)
         (only-in racket/match match-define)
         (only-in threading ~>)
         (only-in "../foreign.rkt" dtype requires-grad? shape tensor?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "layer.rkt" layer? named-parameters))

(provide (contract-out [param-group? (-> any/c boolean?)]))

(module+ private
  (provide floating-params/c
           grouped?
           groups-distinct?
           groups-keys-within?
           groups-setting-holds?
           make-param-group
           param-group-settings
           param-group-tensors
           params/c))

(struct param-group (tensors settings)
  #:constructor-name make-param-group
  #:name param-group-struct) ;; noqa

(define unset (string->uninterned-symbol "unset"))

(define/contract-out hyperparameter-key/c
  contract?
  (flat-named-contract 'hyperparameter-key/c
                       (or/c 'lr 'weight-decay 'momentum 'nesterov? 'beta1 'beta2 'eps
                             'alpha)))

(define/contract-out (param-group params #:lr [lr unset] #:weight-decay [weight-decay unset]
                                  #:momentum [momentum unset] #:nesterov? [nesterov? unset]
                                  #:beta1 [beta1 unset] #:beta2 [beta2 unset]
                                  #:eps [eps unset] #:alpha [alpha unset])
  (->i ([params (listof tensor?)])
       (#:lr [lr (>=/c 0)] #:weight-decay [weight-decay (>=/c 0)]
        #:momentum [momentum (>=/c 0)] #:nesterov? [nesterov? boolean?]
        #:beta1 [beta1 real?] #:beta2 [beta2 real?] #:eps [eps real?]
        #:alpha [alpha (>=/c 0)])
       [result param-group?])
  (make-param-group params
                    (for/hasheq ([k (in-list '(lr weight-decay momentum nesterov? beta1 beta2 eps
                                                  alpha))]
                                 [v (in-list (list lr weight-decay momentum nesterov? beta1 beta2
                                                   eps alpha))]
                                 #:unless (eq? v unset))
                      (values k v))))

(define/contract-out (param-group-params g)
  (-> param-group? (listof tensor?))
  (param-group-tensors g))

(define/contract-out (param-group-hyperparameters g)
  (-> param-group? (hash/c symbol? (or/c real? boolean?) #:immutable #t))
  (param-group-settings g))

(define (last-segment name)
  (regexp-replace #rx"^.*[.]" name ""))

(define/contract-out (weight-decay-param-groups model #:weight-decay [weight-decay #f]
                                                #:skip [skip '("cls-token" "pos-embed")])
  (->* (layer?) (#:weight-decay (or/c #f (>=/c 0)) #:skip (listof string?))
       (list/c param-group? param-group?))
  (define (exempt? named)
    (match-define (cons name t) named)
    (or (~> t shape length (<= 1)) (and (member (last-segment name) skip) #t)))
  (define-values (no-decay decay)
    (partition exempt? (filter (lambda (named) (requires-grad? (cdr named)))
                               (named-parameters model))))
  (list (make-param-group (map cdr decay)
                          (if weight-decay (hasheq 'weight-decay weight-decay) (hasheq)))
        (make-param-group (map cdr no-decay) (hasheq 'weight-decay 0.0))))

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
