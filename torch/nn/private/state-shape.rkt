#lang racket/base

(require (only-in racket/contract/base and/c hash/c hash/dc or/c))

(provide immutable-equal-hash/c
         record/c
         state-real/c
         string-keyed/c)

(define state-real/c (or/c flonum? (and/c rational? exact?)))

(define immutable-equal-hash/c (and/c hash? immutable? hash-equal?))

(define (string-keyed/c value/c)
  (and/c immutable-equal-hash/c (hash/c string? value/c #:flat? #t)))

(define (record/c fields #:required [required (hash-keys fields)])
  (and/c immutable-equal-hash/c
         (hash/dc [key (apply or/c (hash-keys fields))]
                  [value (key) (hash-ref fields key)]
                  #:kind 'flat)
         (lambda (h) (for/and ([k (in-list required)]) (hash-has-key? h k)))))
