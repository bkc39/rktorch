#lang racket/base

(module+ test
  (require (only-in racket/list make-list)
           (only-in racket/match match-define)
           rackunit
           "../main.rkt"
           "../nn.rkt"
           (only-in "../private/step-log.rkt" log-optimizer-step! settle!))

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
      (check-equal? (vector-length e) 4)
      (check-true (flonum? (vector-ref e 1)))
      (check-true (exact-nonnegative-integer? (vector-ref e 2)))
      (check-true (flonum? (vector-ref e 3)))
      (check-true (<= (vector-ref e 3) (vector-ref e 1)) "the host is done before the step ends"))
    (check-true (apply <= (map (lambda (e) (vector-ref e 1)) events))))

  (test-case "the host stamp is taken before a slow settle, not after it"
    (define p (fresh-parameter))
    (define called (current-inexact-monotonic-milliseconds))
    (match-define (list (vector _ ended _ host-done))
      (logged-steps (lambda () (log-optimizer-step! 'slow (make-list 100000 p)))))
    (check-true (< (- host-done called) (- ended host-done))
                "the stamp sits nearer the call than the step's end"))

  (test-case "an empty first parameter neither breaks a logged step nor the device sync"
    (define vacant (Parameter (ones 0)))
    (~> (* vacant vacant) sum backward!)
    (define full (fresh-parameter))
    (define opt (sgd (list vacant full) #:lr 0.1))
    (check-equal? (map (lambda (e) (vector-ref e 0)) (logged-steps (lambda () (step! opt))))
                  (list (eq-hash-code opt)))
    (check-equal? (settle! (list vacant full) #:accelerator-of (lambda (_p) 'cuda:0))
                  (list full))
    (check-equal? (settle! (list vacant) #:accelerator-of (lambda (_p) 'cuda:0)) '()))

  (test-case "settling syncs the first parameter on every accelerator, skipping the CPU"
    (define cpu (fresh-parameter))
    (define gpu0 (fresh-parameter))
    (define gpu0-again (fresh-parameter))
    (define gpu1 (fresh-parameter))
    (define devices (hasheq gpu0 'cuda:0 gpu0-again 'cuda:0 gpu1 'cuda:1))
    (define (accelerator-of p) (hash-ref devices p #f))
    (check-equal? (settle! (list cpu gpu0 cpu gpu0-again gpu1) #:accelerator-of accelerator-of)
                  (list gpu0 gpu1))
    (check-equal? (settle! (list cpu cpu) #:accelerator-of accelerator-of) '())
    (check-equal? (settle! (list (fresh-parameter))) '())))
