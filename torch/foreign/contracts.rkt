#lang racket/base

(require (only-in racket/contract/base
                  ->
                  ->i
                  any/c
                  case->
                  list/c
                  listof
                  none/c
                  or/c
                  unsupplied-arg?)
         (only-in "device-type.rkt" device?)
         (only-in "structs.rkt" tensor?))

(provide dims-rest/c
         index/c
         device/c
         tensor-or-real/c
         pool-size/c
         pos-size/c
         nonneg-size/c
         binary-arith/c
         unary-numeric/c
         log/c
         reduce-or-variadic/c
         argmax/c
         compare/c
         flatten/c
         arange/c)

(define dims-rest/c (listof exact-nonnegative-integer?))

;; -1 means "infer this dimension"
(define index/c exact-integer?)

(define tensor-or-real/c (or/c tensor? real?))

(define device/c
  (or/c device? 'cpu 'cuda 'mps (list/c 'cuda exact-nonnegative-integer?)))

(define binary-arith/c
  (->i ([a (or/c tensor? real?)]
        [b (a) (if (tensor? a) (or/c tensor? real?) tensor?)])
       [result tensor?]))

(define unary-numeric/c
  (->i ([v (or/c tensor? number?)])
       [result (v) (if (tensor? v) tensor? number?)]))

(define log/c
  (->i ([v (or/c tensor? number?)])
       ([base (v) (if (tensor? v) none/c number?)])
       [result (v) (if (tensor? v) tensor? number?)]))

(define reduce-or-variadic/c
  (->i ([v (or/c tensor? real?)])
       #:rest [rest (v) (if (tensor? v) null? (listof real?))]
       [result (v) (if (tensor? v) tensor? real?)]))

(define argmax/c
  (->i ([v (or/c tensor? procedure?)])
       ([dim (v) (if (tensor? v) index/c list?)]
        #:keepdim [keepdim (v) (if (tensor? v) boolean? none/c)])
       #:pre/name (v dim) "a list argument is required with a procedure"
       (or (tensor? v) (not (unsupplied-arg? dim)))
       [result any/c]))

(define pool-size/c (or/c index/c (list/c index/c index/c)))

(define pos-size/c
  (or/c exact-positive-integer?
        (list/c exact-positive-integer? exact-positive-integer?)))
(define nonneg-size/c
  (or/c exact-nonnegative-integer?
        (list/c exact-nonnegative-integer? exact-nonnegative-integer?)))

(define compare/c (-> tensor? tensor-or-real/c tensor?))

(define flatten/c
  (->i ([v any/c])
       ([start (v) (if (tensor? v) index/c none/c)]
        [end (v) (if (tensor? v) index/c none/c)])
       [result any/c]))

(define arange/c
  (case-> (-> real? tensor?)
          (-> real? real? tensor?)
          (-> real? real? real? tensor?)))
