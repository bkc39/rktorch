#lang racket/base

(require (only-in racket/contract/base -> any/c contract-out)
         (only-in "../foreign.rkt" detach requires-grad! tensor?)
         (only-in "../foreign/structs.rkt"
                  tensor-handle tensor-impl tensor-impl-shape)
         (only-in "../private/contract.rkt"
                  define/checked-out define/contract-out))

(provide Parameter? Buffer?)

(struct Parameter% tensor-impl ()
  #:reflection-name 'Parameter)

(struct Buffer% tensor-impl ()
  #:reflection-name 'Buffer)

(define Parameter? Parameter%?)
(define Buffer? Buffer%?)

(module+ checked
  (provide (contract-out [Parameter? (-> any/c boolean?)]
                         [Buffer? (-> any/c boolean?)])))

;; detached first, so a parameter is a leaf whatever graph built its value
(define/checked-out (Parameter t) ;; noqa
  (-> tensor? Parameter?)
  (requires-grad! (Parameter% (tensor-handle (detach t))
                              (tensor-impl-shape t))))

(define/contract-out (Buffer t) ;; noqa
  (-> tensor? Buffer?)
  (Buffer% (tensor-handle (detach t)) (tensor-impl-shape t)))
