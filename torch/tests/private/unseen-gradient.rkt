#lang racket/base

(require (only-in "../../foreign.rkt" cpu-device native-memory-use)
         (only-in "../../foreign/raw/device-queries.rkt" allocator-reading)
         (only-in "../../foreign/raw/pressure.rkt" native-collect-margin))

(provide with-unseen-gradient)

(define mib (* 1024 1024))

(define (cpu-bytes)
  (cond
    [(assoc (cpu-device) (native-memory-use)) => cdr]
    [else 0]))

(define (with-unseen-gradient unseen thunk)
  (parameterize ([allocator-reading (lambda (_dev) (+ (cpu-bytes) (unbox unseen)))]
                 [native-collect-margin (* 1024 1024 mib)])
    (thunk)))
