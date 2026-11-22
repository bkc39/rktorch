#lang racket/base

(module+ test
  (require rackunit
           (only-in "../nn.rkt"
                    Linear chained-scheduler cosine-annealing-lr cosine-annealing-warm-restarts
                    exponential-lr lambda-lr learning-rate linear-lr load-optimizer-state!
                    load-scheduler-state! multi-step-lr one-cycle-lr optimizer-state-dict
                    parameters scheduler-rate scheduler-state-dict sequential-lr sgd step!
                    step-lr))

  (define kinds
    `(("step" ,(lambda (o) (step-lr o #:step-size 2 #:gamma 0.5)))
      ("multi-step" ,(lambda (o) (multi-step-lr o #:milestones '(2 4) #:gamma 0.5)))
      ("exponential" ,(lambda (o) (exponential-lr o #:gamma 0.9)))
      ("cosine" ,(lambda (o) (cosine-annealing-lr o #:t-max 5 #:eta-min 0.001)))
      ("linear" ,(lambda (o) (linear-lr o #:start-factor 0.5 #:total-iters 4)))
      ("one-cycle" ,(lambda (o) (one-cycle-lr o #:max-lr 0.1 #:total-steps 12)))
      ("lambda" ,(lambda (o) (lambda-lr o (lambda (t) (expt 0.95 t)))))
      ("chained" ,(lambda (o) (chained-scheduler (list (step-lr o #:step-size 2 #:gamma 0.5)
                                                       (exponential-lr o #:gamma 0.9)))))
      ("sequential" ,(lambda (o) (sequential-lr o (list (linear-lr o #:start-factor 0.5
                                                                   #:total-iters 2)
                                                        (cosine-annealing-lr o #:t-max 5))
                                                #:milestones '(2))))
      ("warm restarts" ,(lambda (o) (cosine-annealing-warm-restarts o #:t-0 3 #:t-mult 2)))
      ("nested" ,(lambda (o)
                   (sequential-lr o
                                  (list (chained-scheduler
                                         (list (linear-lr o #:start-factor 0.5 #:total-iters 2)
                                               (exponential-lr o #:gamma 0.9)))
                                        (cosine-annealing-warm-restarts o #:t-0 2))
                                  #:milestones '(3))))))

  (define (rates s n)
    (for/list ([_ (in-range n)])
      (step! s)
      (list (scheduler-rate s) (learning-rate s))))

  (test-case "T3-L12: k steps, save, rebuild with another lr, load optimizer then scheduler"
    (for* ([kind (in-list kinds)] [k (in-list '(0 1 3 5))])
      (define name (car kind))
      (define make (cadr kind))
      (define net (Linear 2 1))
      (define opt (sgd (parameters net) #:lr 0.1 #:momentum 0.9))
      (define s (make opt))
      (void (rates s k))
      (define opt-state (optimizer-state-dict opt net))
      (define state (scheduler-state-dict s))
      (define rebuilt-opt (sgd (parameters net) #:lr 0.7 #:momentum 0.9))
      (define rebuilt (make rebuilt-opt))
      (load-optimizer-state! rebuilt-opt net opt-state)
      (load-scheduler-state! rebuilt state)
      (check-equal? (scheduler-state-dict rebuilt) state (format "~a after ~a: the state" name k))
      (check-equal? (rates rebuilt 6) (rates s 6) (format "~a after ~a: the rates" name k))))

  (test-case "T3-L12: a state that does not fit the scheduler raises, leaving it as it was"
    (define (fresh make) (make (sgd (parameters (Linear 2 1)) #:lr 0.1)))
    (define (step-of o) (step-lr o #:step-size 2))
    (define (restarts-of o) (cosine-annealing-warm-restarts o #:t-0 2))
    (define (chain-of o) (chained-scheduler (list (step-of o) (exponential-lr o #:gamma 0.9))))
    (define (check-refused label make state rx)
      (define s (fresh make))
      (define before (scheduler-state-dict s))
      (check-exn rx (lambda () (load-scheduler-state! s state)) label)
      (check-equal? (scheduler-state-dict s) before (format "~a: unchanged" label)))
    (define plain (scheduler-state-dict (fresh step-of)))
    (check-refused "two rates for one group" step-of (hash-set plain "_last_lr" '(0.1 0.1))
                   #rx"_last_lr")
    (check-refused "warm restarts without T_cur" restarts-of plain #rx"T_cur")
    (check-refused "T_cur on a step schedule" step-of (scheduler-state-dict (fresh restarts-of))
                   #rx"T_cur")
    (check-refused "a leaf state into a chain" chain-of plain #rx"schedules")
    (define chained (scheduler-state-dict (fresh chain-of)))
    (check-refused "an inner state that does not fit" chain-of
                   (hash-update chained "_schedulers"
                                (lambda (subs) (list (car subs) (hash-set plain "base_lrs" '()))))
                   #rx"_schedulers\\[1\\].*base_lrs")))
