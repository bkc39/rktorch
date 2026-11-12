#lang racket/base

(require (only-in racket/sequence sequence->list)
         (only-in threading ~>)
         (only-in "../../main.rkt" in-flattened-tensor to))

(provide host)

(define (host t)
  (~> t (to 'cpu) in-flattened-tensor sequence->list))
