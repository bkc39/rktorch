#lang racket/base

(module+ test
  (require rackunit
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in "../main.rkt" ones)
           (only-in "../train/private/types.rkt" prop:trainer trainer-info)
           (only-in "../train.rkt" fit)
           (only-in "private/trainer-kit.rkt" interval-message))

  (struct stand-in ()
    #:property prop:trainer
    (trainer-info '(define-training-step define-optimizers) #t '()))

  (define ten-batches
    (dataloader (tensor-dataset (ones 10 1) (ones 10 1)) #:batch-size 1))

  (define (fit-with-interval interval limit)
    (fit (stand-in) ten-batches #:max-epochs 1 #:logger #f
         #:val-check-interval interval #:limit-train-batches limit))

  (test-case "an integer #:val-check-interval is checked against the limited batches"
    (check-exn (lambda (e)
                 (and (exn:fail:contract? e)
                      (regexp-match? (regexp-quote interval-message) (exn-message e))))
               (lambda () (fit-with-interval 5 2)))
    (check-exn exn:fail:unsupported? (lambda () (fit-with-interval 2 2)))
    (check-exn exn:fail:unsupported? (lambda () (fit-with-interval 5 0.5)))))
