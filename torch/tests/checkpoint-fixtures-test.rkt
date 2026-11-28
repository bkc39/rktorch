#lang racket/base

(module+ test
  (require rackunit
           (only-in "../main.rkt" manual-seed! tensor->list)
           (only-in "../nn.rkt" parameters)
           "../train.rkt"
           (only-in "../train/private/types.rkt" callback-state-key)
           (only-in "private/checkpoint-kit.rkt" fit-draws)
           (only-in "private/resume-harness.rkt" training-world world-callbacks world-train
                    world-trainer world-val))

  (define (values-of layer) (map tensor->list (parameters layer)))

  (test-case "the resume fixture's callbacks construct and run a fit, the average registered"
    (manual-seed! 0)
    (define w (training-world))
    (define averaging (for/first ([cb (in-list (world-callbacks w))]
                                  #:when (equal? (callback-state-key cb) "averaging"))
                        cb))
    (check-equal? (length (parameters averaging)) 4 "the average is a child of its callback")
    (define built (values-of averaging))
    (void (fit (world-trainer w) (world-train w) #:val (world-val w) #:max-epochs 1 #:logger #f
               #:accumulate-grad-batches 2 #:gradient-clip-val 0.5
               #:callbacks (world-callbacks w)))
    (check-not-equal? (values-of averaging) built "setup and the batch ends wrote the average")
    (check-not-equal? (values-of averaging) (values-of (world-trainer w))
                      "the average lags the trained model"))

  (test-case "the drawing fixture fits, and its fit draws from the default stream"
    (define plain (fit-draws '()))
    (check-not-equal? (hash-ref plain 'cpu-rng) (hash-ref plain 'seeded-rng))
    (check-equal? (fit-draws '()) plain "two plain fits draw alike")))
