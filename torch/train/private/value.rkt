#lang racket/base

(require (only-in racket/match match)
         (only-in "../../foreign.rkt" dtype tensor?))

(provide checkpoint-value?)

(define tagged-dtypes '(float32 float64 float16 bfloat16 int64 int32 int16 int8 uint8 bool))

(define (reserved-key? k)
  (and (string? k) (positive? (string-length k)) (char=? (string-ref k 0) #\$)))

(define (exact-rational? x)
  (and (rational? x) (exact? x)))

(define (taggable-hash? h)
  (and (or (hash-equal? h) (hash-eqv? h) (hash-eq? h))
       (not (hash-weak? h))
       (not (hash-ephemeron? h))))

(define (checkpoint-value? v)
  (define on-path (make-hasheq))
  (define accepted (make-hasheq))
  (define (node? v walk)
    (cond
      [(hash-ref accepted v #f) #t]
      [(hash-ref on-path v #f) #f]
      [else
       (hash-set! on-path v #t)
       (define ok? (walk))
       (hash-remove! on-path v)
       (when ok? (hash-set! accepted v #t))
       ok?]))
  (let value? ([v v])
    (match v
      [(or #t #f '()) #t]
      [(or (? flonum?) (? string?) (? exact-rational?)) #t]
      [(? symbol?) (symbol-interned? v)]
      [(? tensor?) (and (memq (dtype v) tagged-dtypes) #t)]
      [(cons a d) (node? v (lambda () (and (value? a) (value? d))))]
      [(? hash?)
       (and (taggable-hash? v)
            (node? v (lambda ()
                       (for/and ([(k x) (in-hash v)])
                         (and (not (reserved-key? k)) (value? k) (value? x))))))]
      [_ #f])))
