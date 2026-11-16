#lang racket/base

(require (only-in racket/list remove-duplicates)
         (only-in racket/sequence sequence->list)
         (only-in threading ~>)
         (only-in "../../main.rkt" in-flattened-tensor tensor->bytes to))

(provide bytes-of
         distinct-runs
         host)

(define (host t)
  (~> t (to 'cpu) in-flattened-tensor sequence->list))

(define (bytes-of t)
  (~> t (to 'cpu) tensor->bytes))

(define (distinct-runs n run)
  (~> (for/list ([_ (in-range n)]) (run)) remove-duplicates length))
