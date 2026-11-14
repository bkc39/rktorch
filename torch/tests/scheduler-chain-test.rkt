#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           (only-in "../main.rkt" zeros)
           (only-in "../nn.rkt"
                    Parameter chained-scheduler cosine-annealing-lr cosine-annealing-warm-restarts
                    exponential-lr lambda-lr learning-rate linear-lr param-group-ref
                    param-group-set! scheduler-rate scheduler-step-count sequential-lr
                    set-learning-rate! sgd step! step-lr)
           (only-in (submod "../foreign/raw/cas.rkt" private) site-cas)
           (only-in "../foreign/raw/regions.rkt" with-cleanup)
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/scheduler-steps.rkt" stepped-reads two-group-opt)
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

  (bounded-test "a sequence switches to its second schedule's step 0 at the milestone"
    (define opt (fresh-opt 1.0))
    (define warm (linear-lr opt #:start-factor 0.25 #:total-iters 4))
    (define cool (cosine-annealing-lr opt #:t-max 8 #:eta-min 0.01))
    (define s (sequential-lr opt (list warm cool) #:milestones '(4)))
    (check-equal? (stepped-reads s scheduler-rate 6)
                  '(0.25 0.4375 0.625 0.8125 1.0 0.9623203685930869 0.855017856687341)))

  (bounded-test "unsorted milestones pick the schedule as torch's bisect_right does"
    (define opt (fresh-opt 1.0))
    (define s (sequential-lr opt (list (exponential-lr opt #:gamma 0.5)
                                       (step-lr opt #:step-size 100)
                                       (linear-lr opt #:start-factor 0.5 #:total-iters 2))
                             #:milestones '(4 2)))
    (check-equal? (stepped-reads s scheduler-rate 6) '(1.0 0.5 0.5 0.75 1.0 1.0 1.0)))

  (bounded-test "a chain a sequence starts with is at step index 0, as its schedules are"
    (define opt (fresh-opt 1.0))
    (define exp-decay (exponential-lr opt #:gamma 0.9))
    (define chain (chained-scheduler (list exp-decay (step-lr opt #:step-size 3))))
    (define s (sequential-lr opt (list chain (step-lr opt #:step-size 2)) #:milestones '(3)))
    (check-equal? (map scheduler-step-count (list chain exp-decay s)) '(0 0 0))
    (step! s)
    (check-equal? (map scheduler-step-count (list chain exp-decay s)) '(1 1 1)))

  (bounded-test "a chain counts its own steps from 0, whatever its schedules had reached"
    (define opt (fresh-opt 1.0))
    (define exp-decay (exponential-lr opt #:gamma 0.9))
    (for ([_ (in-range 3)]) (step! exp-decay))
    (define chain (chained-scheduler (list exp-decay (step-lr opt #:step-size 2))))
    (check-equal? (map scheduler-step-count (list chain exp-decay)) '(0 3)))

  (define (with-write-at-first-update write thunk)
    (define primitive (unbox site-cas))
    (define pending? (box #t))
    (define (racing site target old new)
      (when (and (eq? site 'update) (unbox pending?))
        (set-box! pending? #f)
        (write))
      (primitive site target old new))
    (set-box! site-cas racing)
    (with-cleanup (thunk) #:finally (set-box! site-cas primitive)))

  (bounded-test "a rate written while a schedule steps is scaled by the step, not lost"
    (define opt (fresh-opt 1.0))
    (define s (exponential-lr opt #:gamma 0.5))
    (with-write-at-first-update (lambda () (set-learning-rate! opt 10.0)) (lambda () (step! s)))
    (check-not-false (memv (learning-rate opt) '(5.0 10.0)))
    (check-equal? (scheduler-rate s) (learning-rate opt)))

  (bounded-test "a lambda-lr factor that writes the optimizer runs once per group and step"
    (define opt (two-group-opt))
    (define calls (box 0))
    (define (factor t)
      (set-box! calls (add1 (unbox calls)))
      (when (= t 1) (param-group-set! opt 0 'weight-decay 0.1))
      1.0)
    (define s (lambda-lr opt factor))
    (define groups 2)
    (define steps 3)
    (for ([_ (in-range steps)]) (step! s))
    (check-equal? (unbox calls) (* groups (add1 steps)))
    (check-equal? (learning-rate opt) 1.0)
    (check-equal? (param-group-ref opt 0 'weight-decay) 0.1))

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
