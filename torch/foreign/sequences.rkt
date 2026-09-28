#lang racket/base

(require (only-in racket/contract/base -> flat-named-contract)
         (only-in "../generated.rkt" select-int)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "ops.rkt" tensor->list tensor-shape)
         (only-in "structs.rkt" tensor?))

(define iterable-tensor/c
  (flat-named-contract
   'tensor-of-rank-at-least-one
   (lambda (t) (and (tensor? t) (pair? (tensor-shape t))))))

(define/contract-out (in-tensor t) ;; noqa
  (-> iterable-tensor/c sequence?)
  (define n (car (tensor-shape t)))
  (make-do-sequence
   (lambda ()
     (values (lambda (i) (select-int t 0 i))
             add1
             0
             (lambda (i) (< i n))
             #f
             #f))))

(define/contract-out (in-flattened-tensor t) ;; noqa
  (-> tensor? sequence?)
  (make-do-sequence
   (lambda ()
     (values car cdr (tensor->list t) pair? #f #f))))
