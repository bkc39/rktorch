#lang racket/base

(require (only-in racket/match match)
         (only-in "../../main.rkt" dtype tensor->bytes tensor-shape tensor? to-device)
         (only-in "../../nn.rkt" state-dict))

(provide comparable host-bytes layer-bytes)

(define (host-bytes t)
  (list (dtype t) (tensor-shape t) (tensor->bytes (to-device t 'cpu))))

(define (comparable v #:hash-kinds? [kinds? #f])
  (let walk ([v v])
    (match v
      [(? tensor?) (host-bytes v)]
      [(? hash?)
       (define entries (for/list ([(k x) (in-hash v)]) (cons (walk k) (walk x))))
       (cond
         [(not kinds?) (make-immutable-hash entries)]
         [(hash-eq? v) (make-immutable-hasheq entries)]
         [(hash-eqv? v) (make-immutable-hasheqv entries)]
         [else (make-immutable-hash entries)])]
      [(cons a d) (cons (walk a) (walk d))]
      [_ v])))

(define (layer-bytes m)
  (comparable (state-dict m)))
