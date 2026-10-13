#lang racket/base

(require (only-in racket/base
                  [+ base:+]
                  [- base:-]
                  [* base:*]
                  [/ base:/]
                  [< base:<]
                  [<= base:<=]
                  [> base:>]
                  [>= base:>=])
         (only-in "promoted.rkt" ge gt le lt)
         (only-in "raw/syntax.rkt" define-arith)
         (only-in "structs.rkt" tensor?)
         (only-in "tensor-ops.rkt" add div matmul mul neg sub))

(provide t+ t- t* t/ t< t<= t> t>= @)

(define-arith t+ tensor? add base:+ values)
(define-arith t- tensor? sub base:- neg)
(define-arith t* tensor? mul base:* values)
(define-arith t/ tensor? div base:/ (lambda (t) (div 1.0 t)))

(define (ordering name tensor-op mirror-op base-op)
  (case-lambda
    [(a b)
     (cond
       [(tensor? a) (tensor-op a b)]
       [(tensor? b) (mirror-op b a)]
       [else (base-op a b)])]
    [args
     (when (ormap tensor? args)
       (raise-arguments-error name "a tensor comparison takes two operands"
                              "operands" args))
     (apply base-op args)]))

(define t< (ordering '< lt gt base:<))
(define t<= (ordering '<= le ge base:<=))
(define t> (ordering '> gt lt base:>))
(define t>= (ordering '>= ge le base:>=))

(define (@ a . rest)
  (foldl (lambda (b acc) (matmul acc b)) a rest))
