#lang racket/base

(require (only-in racket/base [length list-length])
         (only-in racket/contract/base -> any/c contract-out)
         (only-in racket/generic define-generics)
         (only-in "ops.rkt" tensor-shape)
         (only-in "structs.rkt" tensor?))

;; the noqa'd exports are macro expansions raco review cannot see
(provide gen:sized
         (contract-out
          [sized? (-> any/c boolean?)]
          [length (-> sized? exact-nonnegative-integer?)])) ;; noqa

;; Python's len: one generic over the core containers, tensors along
;; their first dimension, and whatever else answers gen:sized
(define-generics sized
  (length sized)
  #:fast-defaults
  ([list? (define length list-length)] ;; noqa
   [vector? (define length vector-length)] ;; noqa
   [string? (define length string-length)] ;; noqa
   [hash? (define length hash-count)]) ;; noqa
  #:defaults
  ([tensor?
    (define (length t) ;; noqa
      (define dims (tensor-shape t))
      (when (null? dims)
        (raise-argument-error 'length "a tensor of rank at least one" t))
      (car dims))]))
