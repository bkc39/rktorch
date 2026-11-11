#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           (only-in "../main.rkt" zeros)
           (only-in "../nn.rkt"
                    Parameter chained-scheduler cosine-annealing-lr cosine-annealing-warm-restarts
                    exponential-lr learning-rate linear-lr scheduler-rate sequential-lr sgd step!
                    step-lr)
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/within.rkt" bounded-test))

  (define blames-this-test? (blames-caller? "scheduler-chain-test"))

  (define (fresh-opt [lr 0.1])
    (sgd (list (Parameter (zeros 1))) #:lr lr))

  (define (relatively-close? got want)
    (<= (abs (- got want)) (* 1e-15 (abs want))))

  (bounded-test "G2-L3: a chain of exponential and step decays multiplies their factors"
    (define opt (fresh-opt 0.1))
    (define chain (chained-scheduler (list (exponential-lr opt #:gamma 0.9)
                                           (step-lr opt #:step-size 3 #:gamma 0.5))))
    (for ([t (in-range 1 13)])
      (step! chain)
      (define want (* 0.1 (expt 0.9 t) (expt 0.5 (quotient t 3))))
      (check-true (relatively-close? (learning-rate opt) want) (format "step ~a" t))
      (check-true (relatively-close? (scheduler-rate chain) want) (format "step ~a" t))))

  (bounded-test "G2-L3: a sequence restarts its second schedule from step 0 at the milestone"
    (define opt (fresh-opt 1.0))
    (define warm (linear-lr opt #:start-factor 0.25 #:total-iters 4))
    (define cool (cosine-annealing-lr opt #:t-max 8 #:eta-min 0.01))
    (define s (sequential-lr opt (list warm cool) #:milestones '(4)))
    (check-= (learning-rate opt) 0.25 1e-15)
    (for ([_ (in-range 4)]) (step! s))
    (check-= (learning-rate opt) 1.0 1e-15))

  (bounded-test "G2-L3: warm restarts return to the base rate at each period's start"
    (define opt (fresh-opt 0.1))
    (define s (cosine-annealing-warm-restarts opt #:t-0 3 #:t-mult 2 #:eta-min 0.01))
    (define rates (for/list ([_ (in-range 9)]) (step! s) (learning-rate opt)))
    (check-= (list-ref rates 2) 0.1 1e-15)
    (check-= (list-ref rates 8) 0.1 1e-15))

  (define (two-optimizers)
    (values (fresh-opt) (fresh-opt)))

  (define bad-calls
    `(("sequential-lr with one milestone too many"
       ,(lambda ()
          (define opt (fresh-opt))
          (sequential-lr opt (list (exponential-lr opt #:gamma 0.9)) #:milestones '(2))))
      ("sequential-lr over a schedule of another optimizer"
       ,(lambda ()
          (define-values (a b) (two-optimizers))
          (sequential-lr a (list (exponential-lr a #:gamma 0.9) (exponential-lr b #:gamma 0.9))
                         #:milestones '(2))))
      ("sequential-lr given a schedule as its optimizer"
       ,(lambda ()
          (define opt (fresh-opt))
          (define inner (exponential-lr opt #:gamma 0.9))
          (sequential-lr inner (list (step-lr opt #:step-size 2)) #:milestones '())))
      ("chained-scheduler over two optimizers"
       ,(lambda ()
          (define-values (a b) (two-optimizers))
          (chained-scheduler (list (exponential-lr a #:gamma 0.9)
                                   (exponential-lr b #:gamma 0.9)))))))

  (for ([case (in-list bad-calls)])
    (match-define (list label call) case)
    (bounded-test (format "G2-L4: ~a blames the caller" label)
      (check-exn blames-this-test? call))))
