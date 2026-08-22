#lang racket/base

(require (only-in racket/base
                  [+ base:+]
                  [- base:-]
                  [* base:*]
                  [/ base:/])
         (only-in "raw/syntax.rkt" define-arith)
         (only-in "structs.rkt" tensor?)
         (only-in "tensor-ops.rkt" add div matmul mul neg sub))

(provide t+ t- t* t/ @)

(define-arith t+ tensor? add base:+ values)
(define-arith t- tensor? sub base:- neg)
(define-arith t* tensor? mul base:* values)
(define-arith t/ tensor? div base:/ (lambda (t) (div 1.0 t)))

(define (@ a . rest)
  (foldl (lambda (b acc) (matmul acc b)) a rest))
