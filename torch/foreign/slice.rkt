#lang racket/base

(require (only-in racket/contract/base -> any/c case-> contract-out or/c)
         (only-in "../private/contract.rkt" define/checked-out))

(provide slice? slice-start slice-end slice-step)

(struct slice (start end step) #:transparent)

(module+ checked
  (provide (contract-out [slice? (-> any/c boolean?)])))

(define bound/c (or/c #f exact-integer?))

(define/checked-out :: ;; noqa
  (case-> (-> slice?)
          (-> bound/c slice?)
          (-> bound/c bound/c slice?)
          (-> bound/c bound/c exact-integer? slice?))
  (case-lambda
    [() (slice #f #f 1)]
    [(end) (slice #f end 1)]
    [(start end) (slice start end 1)]
    [(start end step) (slice start end step)]))
