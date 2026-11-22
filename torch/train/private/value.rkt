#lang racket/base

(require (only-in racket/match match)
         (only-in "../../foreign.rkt" dtype tensor?))

(provide checkpoint-value?)

(define tagged-dtypes '(float32 float64 float16 bfloat16 int64 int32 int16 int8 uint8 bool))

(define (reserved-key? k)
  (and (string? k) (positive? (string-length k)) (char=? (string-ref k 0) #\$)))

(define (checkpoint-value? v)
  (let value? ([v v] [open (hasheq)])
    (match v
      [(or #t #f '()) #t]
      [(or (? exact-integer?) (? flonum?) (? string?)) #t]
      [(? symbol?) (symbol-interned? v)]
      [(? tensor?) (and (memq (dtype v) tagged-dtypes) #t)]
      [(cons a d) (and (value? a open) (value? d open))]
      [(? hash?)
       (and (not (hash-ref open v #f))
            (let ([open (hash-set open v #t)])
              (for/and ([(k x) (in-hash v)])
                (and (not (reserved-key? k)) (value? k open) (value? x open)))))]
      [_ #f])))
