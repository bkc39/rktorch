#lang racket/base

(module+ test
  (require rackunit
           "../main.rkt"
           "../nn.rkt")

  (define (logged-steps thunk)
    (define receiver (make-log-receiver (current-logger) 'debug 'rktorch-step))
    (thunk)
    (let drain ([events '()])
      (define event (sync/timeout 0 receiver))
      (if event (drain (cons (vector-ref event 2) events)) (reverse events))))

  (define (fresh-parameter)
    (define p (Parameter (ones 2)))
    (backward! (sum (mul p p)))
    p)

  (test-case "every optimizer logs its own steps, a schedule none"
    (define opts
      (list (sgd (list (fresh-parameter)) #:lr 0.1)
            (adam (list (fresh-parameter)) #:lr 0.1)
            (rmsprop (list (fresh-parameter)) #:lr 0.1)))
    (define schedule (step-lr (car opts) #:step-size 1))
    (define events
      (logged-steps
       (lambda ()
         (for ([opt (in-list opts)]) (step! opt))
         (step! schedule))))
    (check-equal? (map (lambda (e) (vector-ref e 0)) events)
                  (map eq-hash-code opts))
    (for ([e (in-list events)])
      (check-true (flonum? (vector-ref e 1)))
      (check-true (exact-nonnegative-integer? (vector-ref e 2))))
    (check-true (apply <= (map (lambda (e) (vector-ref e 1)) events)))))
