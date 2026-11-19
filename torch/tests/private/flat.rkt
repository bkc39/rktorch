#lang racket/base

(require (only-in threading ~>)
         (only-in "../../main.rkt" reshape tensor->list))

(provide flat)

(define (flat t)
  (~> t (reshape -1) tensor->list))
