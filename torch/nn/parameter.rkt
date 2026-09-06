#lang racket/base

(require (only-in racket/contract/base -> any/c) ;; noqa
         (only-in "../foreign.rkt" detach requires-grad! tensor?)
         (only-in "../foreign/structs.rkt"
                  tensor-handle tensor-impl tensor-impl-shape)
         (only-in "../private/contract.rkt" define/checked-out))

(struct Parameter% tensor-impl ()
  #:reflection-name 'Parameter)

(define/checked-out Parameter? (-> any/c boolean?) Parameter%?)

(define/checked-out (Parameter t) ;; noqa
  (-> tensor? Parameter?)
  (requires-grad! (Parameter% (tensor-handle (detach t))
                              (tensor-impl-shape t))))
