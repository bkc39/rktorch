#lang racket/base

(provide lbfgs-minimize!)

;; one call is one torch.optim.LBFGS step over its closure, with torch's defaults
(define (lbfgs-minimize! params closure ;; noqa
                         #:lr [lr 1.0]
                         #:max-iter [max-iter 20]
                         #:max-eval [max-eval #f]
                         #:tolerance-grad [tolerance-grad 1e-7]
                         #:tolerance-change [tolerance-change 1e-9]
                         #:history-size [history-size 100]
                         #:line-search [line-search #f])
  (error 'unimplemented))
