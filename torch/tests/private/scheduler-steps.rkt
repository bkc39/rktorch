#lang racket/base

(require (only-in "../../main.rkt" zeros)
         (only-in "../../nn.rkt" Parameter param-group sgd step!))

(provide stepped-reads
         two-group-opt)

(define (stepped-reads s probe steps)
  (cons (probe s) (for/list ([_ (in-range steps)]) (step! s) (probe s))))

(define (two-group-opt #:lr [lrs '(1.0 1.0)])
  (sgd (for/list ([lr (in-list lrs)]) (param-group (list (Parameter (zeros 1))) #:lr lr))
       #:lr (car lrs)))
