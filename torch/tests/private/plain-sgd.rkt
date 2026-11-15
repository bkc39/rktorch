#lang racket/base

(require (only-in "../../foreign.rkt" maybe-grad sub! with-no-grad)
         (only-in "../../nn/optim.rkt" gen:optimizer))

(provide plain-sgd)

(struct plain-sgd (params lr)
  #:methods gen:optimizer
  [(define (optimizer-parameters opt) (plain-sgd-params opt))
   (define (optimizer-step! opt)
     (with-no-grad
       (for* ([p (in-list (plain-sgd-params opt))]
              [g (in-value (maybe-grad p))]
              #:when g)
         (sub! p g (plain-sgd-lr opt)))))])
