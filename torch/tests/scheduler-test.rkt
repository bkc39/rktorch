#lang racket/base

(module+ test
  (require rackunit
           "../main.rkt"
           "../nn.rkt")

  (define (fresh-opt [lr 0.1])
    (sgd (list (Parameter (zeros 1))) #:lr lr))

  ;; the rate at construction, then after each of n steps
  (define (rates s n)
    (cons (learning-rate s)
          (for/list ([_ (in-range n)])
            (step! s)
            (learning-rate s))))

  (define (check-rates got want)
    (for ([g (in-list got)] [w (in-list want)] [i (in-naturals)])
      (check-= g w 1e-9 (format "step ~a" i))))

  (test-case "a scheduler answers to the optimizer protocol"
    (define opt (fresh-opt))
    (define s (step-lr opt #:step-size 2 #:gamma 0.5))
    (check-true (scheduler? s))
    (check-true (optimizer? s))
    (check-false (scheduler? opt))
    (check-eq? (scheduler-optimizer-of s) opt)
    (check-equal? (scheduler-step-count s) 0)
    (check-= (scheduler-rate s) 0.1 0.0)
    (check-= (learning-rate opt) 0.1 0.0 "construction writes the rate")
    (step! s)
    (check-equal? (scheduler-step-count s) 1)
    (zero-grads! s)
    (check-= (learning-rate s) (learning-rate opt) 0.0
             "the schedule reports its optimizer's rate"))

  (test-case "step-lr halves every two steps"
    (check-rates (rates (step-lr (fresh-opt) #:step-size 2 #:gamma 0.5) 5)
                 '(0.1 0.1 0.05 0.05 0.025 0.025)))

  (test-case "multi-step-lr decays at each milestone"
    (check-rates (rates (multi-step-lr (fresh-opt) #:milestones '(1 3)) 4)
                 '(0.1 0.01 0.01 0.001 0.001)))

  (test-case "exponential-lr multiplies by gamma each step"
    (check-rates (rates (exponential-lr (fresh-opt) #:gamma 0.5) 3)
                 '(0.1 0.05 0.025 0.0125)))

  (test-case "cosine-annealing-lr reaches eta-min at t-max"
    (define got (rates (cosine-annealing-lr (fresh-opt) #:t-max 4
                                            #:eta-min 0.02)
                       4))
    (check-= (car got) 0.1 1e-12)
    (check-= (list-ref got 2) 0.06 1e-12 "halfway is the midpoint")
    (check-= (list-ref got 4) 0.02 1e-12))

  (test-case "linear-lr warms up from the start factor over total-iters"
    (check-rates (rates (linear-lr (fresh-opt) #:start-factor 0.25
                                   #:end-factor 1.0 #:total-iters 3)
                        4)
                 '(0.025 0.05 0.075 0.1 0.1)))

  (test-case "one-cycle-lr rises to max-lr, falls to the final rate, then refuses"
    (define s (one-cycle-lr (fresh-opt) #:max-lr 1.0 #:total-steps 10
                            #:pct-start 0.3 #:div-factor 10
                            #:final-div-factor 100))
    (define got (rates s 9))
    (check-= (car got) 0.1 1e-12 "max-lr over div-factor")
    (check-= (list-ref got 2) 1.0 1e-9 "the peak at pct-start times total")
    (check-= (list-ref got 9) 0.001 1e-9 "the final rate")
    (check-true (< (car got) (cadr got) (caddr got)) "rising")
    ;; torch allows exactly total-steps steps; the one after raises
    (step! s)
    (check-exn #rx"stepped past the cycle" (lambda () (step! s))))

  (test-case "lambda-lr scales the base rate by the function of the step"
    (check-rates (rates (lambda-lr (fresh-opt) (lambda (t) (/ 1.0 (add1 t))))
                        3)
                 '(0.1 0.05 0.03333333333333333 0.025)))

  (test-case "set-learning-rate! and a scheduler on adam and rmsprop"
    (define a (adam (list (Parameter (zeros 2))) #:lr 0.01))
    (set-learning-rate! a 0.5)
    (check-= (learning-rate a) 0.5 0.0)
    (define r (rmsprop (list (Parameter (zeros 2)))))
    (check-= (learning-rate r) 0.01 0.0)
    (define s (exponential-lr r #:gamma 0.1))
    (step! s)
    (check-= (learning-rate r) 0.001 1e-15)
    (check-exn exn:fail:contract?
               (lambda () (one-cycle-lr a #:max-lr 0 #:total-steps 3)))))
