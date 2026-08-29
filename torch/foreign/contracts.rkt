#lang racket/base

(require (only-in racket/contract/base
                  ->
                  ->i
                  and/c
                  any/c
                  case->
                  list/c
                  listof
                  none/c
                  or/c
                  unsupplied-arg?
                  vectorof)
         (only-in "device-type.rkt" device?)
         (only-in "ops.rkt" tensor-dtype tensor-shape)
         (only-in "promoted.rkt" slice-end slice-start slice-step slice?)
         (only-in "structs.rkt" tensor?))

(provide bool-tensor/c
         dims-rest/c
         index-spec/c
         index/c
         index-vector/c
         int64-tensor/c
         device/c
         tensor-or-real/c
         pool-size/c
         size-1d/c
         pos-size-1d/c
         nonneg-size-1d/c
         pos-size/c
         nonneg-size/c
         binary-arith/c
         unary-numeric/c
         unary-real/c
         log/c
         reduce-or-variadic/c
         argmax/c
         compare/c
         flatten/c
         arange/c)

(define dims-rest/c (listof exact-nonnegative-integer?))

(define (slice-spec? x)
  (define (bound? b) (or (not b) (exact-integer? b)))
  (and (slice? x)
       (bound? (slice-start x))
       (bound? (slice-end x))
       (exact-integer? (slice-step x))))

(define bool-tensor/c
  (and/c tensor? (lambda (x) (eq? (tensor-dtype x) 'bool))))

(define int64-tensor/c
  (and/c tensor? (lambda (x) (eq? (tensor-dtype x) 'int64))))

(define index-vector/c
  (and/c int64-tensor/c (lambda (x) (< (length (tensor-shape x)) 2))))

(define (index-tensor-spec? x)
  (and (tensor? x)
       (pair? (tensor-shape x))
       (or (eq? (tensor-dtype x) 'bool)
           (and (eq? (tensor-dtype x) 'int64)
                (= 1 (length (tensor-shape x)))))))

(define index-spec/c
  (or/c exact-integer? slice-spec? index-tensor-spec? #f '...
        (listof exact-integer?) (vectorof exact-integer?)))

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

(define unary-real/c
  (->i ([v (or/c tensor? real?)])
       [result (v) (if (tensor? v) tensor? real?)]))

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

(define size-1d/c (or/c index/c (list/c index/c)))

(define pos-size-1d/c
  (or/c exact-positive-integer? (list/c exact-positive-integer?)))

(define nonneg-size-1d/c
  (or/c exact-nonnegative-integer? (list/c exact-nonnegative-integer?)))

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
