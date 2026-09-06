#lang racket/base

(require (only-in racket/contract/base -> any/c)
         (only-in "../foreign.rkt" detach tensor?)
         (only-in "../foreign/structs.rkt"
                  tensor-handle tensor-impl tensor-impl-shape)
         (only-in "../private/contract.rkt"
                  define/checked-out define/contract-out))

(struct Buffer% tensor-impl ()
  #:reflection-name 'Buffer)

(define/checked-out Buffer? (-> any/c boolean?) Buffer%?)

(define/contract-out (Buffer t) ;; noqa
  (-> tensor? Buffer?)
  (Buffer% (tensor-handle (detach t)) (tensor-impl-shape t)))
