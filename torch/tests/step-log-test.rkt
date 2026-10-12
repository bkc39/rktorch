#lang racket/base

(module+ test
  (require rackunit
           "../main.rkt"
           "../nn.rkt"
           (only-in "../private/step-log.rkt" settle!))

  (define (logged-steps thunk)
    (define receiver (make-log-receiver (current-logger) 'debug 'rktorch-step))
    (thunk)
    (let drain ([events '()])
      (define event (sync/timeout 0 receiver))
      (if event (drain (cons (vector-ref event 2) events)) (reverse events))))

  (define (fresh-parameter)
    (define p (Parameter (ones 2)))
    (~> (* p p) sum backward!)
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
    (check-true (apply <= (map (lambda (e) (vector-ref e 1)) events))))

  (test-case "an empty first parameter neither breaks a logged step nor the device sync"
    (define vacant (Parameter (ones 0)))
    (~> (* vacant vacant) sum backward!)
    (define full (fresh-parameter))
    (define opt (sgd (list vacant full) #:lr 0.1))
    (check-equal? (map (lambda (e) (vector-ref e 0)) (logged-steps (lambda () (step! opt))))
                  (list (eq-hash-code opt)))
    (define seen '())
    (define (on-accelerator? p)
      (set! seen (cons p seen))
      #t)
    (check-not-exn (lambda () (settle! (list vacant full) #:accelerator? on-accelerator?)))
    (check-equal? seen (list full))
    (set! seen '())
    (check-equal? (settle! (list vacant) #:accelerator? on-accelerator?) (void))
    (check-equal? seen '())))
