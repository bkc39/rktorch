#lang racket/base

(module+ test
  (require rackunit
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in "../main.rkt" ones)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           (only-in "../train.rkt"
                    define-init define-optimizers define-trainer define-training-step fit
                    run?)
           (only-in "private/trainer-kit.rkt" interval-message))

  (define-trainer Line (net)
    (define-init () (set! net (Linear 1 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y) (mse-loss (net x) y)))

  (define ten-batches
    (dataloader (tensor-dataset (ones 10 1) (ones 10 1)) #:batch-size 1))

  (define (fit-with-interval interval limit)
    (fit (Line) ten-batches #:max-epochs 1 #:logger #f
         #:val-check-interval interval #:limit-train-batches limit))

  (test-case "an integer #:val-check-interval is checked against the limited batches"
    (check-exn (lambda (e)
                 (and (exn:fail:contract? e)
                      (regexp-match? (regexp-quote interval-message) (exn-message e))))
               (lambda () (fit-with-interval 5 2)))
    (check-pred run? (fit-with-interval 2 2))
    (check-pred run? (fit-with-interval 5 0.5))))
