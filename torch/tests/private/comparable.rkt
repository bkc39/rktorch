#lang racket/base

(require (only-in racket/match match)
         (only-in "../../main.rkt" dtype tensor->bytes tensor-shape tensor? to-device)
         (only-in "../../nn.rkt" state-dict))

(provide comparable host-bytes layer-bytes)

(define (host-bytes t)
  (list (dtype t) (tensor-shape t) (tensor->bytes (to-device t 'cpu))))

(define (comparable v)
  (match v
    [(? tensor?) (host-bytes v)]
    [(? hash?) (for/hash ([(k x) (in-hash v)]) (values (comparable k) (comparable x)))]
    [(cons a d) (cons (comparable a) (comparable d))]
    [_ v]))

(define (layer-bytes m)
  (comparable (state-dict m)))
