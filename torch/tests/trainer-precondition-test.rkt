#lang racket/base

(module+ test
  (require rackunit
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in "../main.rkt" ones)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           (only-in "../train.rkt"
                    define-hook define-init define-optimizers define-trainer
                    define-training-step define-validation-step fit run run-global-step)
           (only-in "private/trainer-kit.rkt" interval-message))

  (define-trainer Line (net validated)
    (define-init (seen) (set! net (Linear 1 1)) (set! validated seen))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-validation-step (x y) (mse-loss (net x) y))
    (define-hook (on-validation-start)
      (set-box! validated (cons (run-global-step run) (unbox validated)))))

  (define ten-batches
    (dataloader (tensor-dataset (ones 10 1) (ones 10 1)) #:batch-size 1))

  (define (validated-steps interval limit)
    (define seen (box '()))
    (fit (Line seen) ten-batches #:val ten-batches #:max-epochs 1 #:logger #f
         #:num-sanity-val-steps 0 #:limit-val-batches 1
         #:val-check-interval interval #:limit-train-batches limit)
    (reverse (unbox seen)))

  (test-case "an integer #:val-check-interval is checked against the limited batches"
    (check-exn (lambda (e)
                 (and (exn:fail:contract? e)
                      (regexp-match? (regexp-quote interval-message) (exn-message e))))
               (lambda () (validated-steps 5 2)))
    (check-equal? (validated-steps 2 2) '(2))
    (check-equal? (validated-steps 1 2) '(1 2))
    (check-equal? (validated-steps 5 0.5) '(5))))
