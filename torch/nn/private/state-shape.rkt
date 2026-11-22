#lang racket/base

(require (only-in racket/contract/base and/c hash/c hash/dc or/c))

(provide record/c
         state-real/c
         string-keyed/c)

(define state-real/c (or/c flonum? (and/c rational? exact?)))

(define (string-keyed/c value/c)
  (and/c hash? immutable? hash-equal? (hash/c string? value/c #:flat? #t)))

(define (record/c fields #:required [required (hash-keys fields)])
  (and/c hash?
         immutable?
         hash-equal?
         (hash/dc [key (apply or/c (hash-keys fields))]
                  [value (key) (hash-ref fields key)]
                  #:kind 'flat)
         (lambda (h) (for/and ([k (in-list required)]) (hash-has-key? h k)))))
