#lang racket/base

(require (only-in racket/match match)
         (only-in "../../main.rkt" dtype tensor->bytes tensor-shape tensor? to-device)
         (only-in "../../nn.rkt" state-dict))

(provide comparable host-bytes layer-bytes)

(define (host-bytes t)
  (list (dtype t) (tensor-shape t) (tensor->bytes (to-device t 'cpu))))

(define (hash-kind h)
  (cond
    [(hash-eq? h) 'eq]
    [(hash-eqv? h) 'eqv]
    [else 'equal]))

(define (comparable v #:hash-kinds? [kinds? #f])
  (let walk ([v v])
    (match v
      [(? tensor?) (host-bytes v)]
      [(? hash?)
       (define entries (for/hash ([(k x) (in-hash v)]) (values (walk k) (walk x))))
       (if kinds? (vector (hash-kind v) entries) entries)]
      [(cons a d) (cons (walk a) (walk d))]
      [_ v])))

(define (layer-bytes m)
  (comparable (state-dict m)))
