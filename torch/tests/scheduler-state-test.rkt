#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           (only-in "../nn.rkt"
                    Linear chained-scheduler cosine-annealing-lr cosine-annealing-warm-restarts
                    exponential-lr lambda-lr learning-rate linear-lr load-optimizer-state!
                    load-scheduler-state! multi-step-lr one-cycle-lr optimizer-state-dict
                    parameters scheduler-rate scheduler-state-dict sequential-lr sgd step!
                    step-lr)
           (only-in "private/comparable.rkt" comparable))

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
      (match-define (list name make) kind)
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
      (check-equal? (comparable (optimizer-state-dict rebuilt-opt net)) (comparable opt-state)
                    (format "~a after ~a: the optimizer, initial_lr included" name k))
      (check-equal? (scheduler-state-dict rebuilt) state (format "~a after ~a: the state" name k))
      (check-equal? (rates rebuilt 6) (rates s 6) (format "~a after ~a: the rates" name k))))

  (test-case "T3-L12: an exact one-cycle base survives a state round trip exact"
    (define net (Linear 2 1))
    (define (cycle o) (one-cycle-lr o #:max-lr 1/10 #:div-factor 25 #:total-steps 12))
    (define opt (sgd (parameters net) #:lr 0.1))
    (define s (cycle opt))
    (void (rates s 3))
    (define state (scheduler-state-dict s))
    (check-equal? (hash-ref state "base_lrs") '(1/250))
    (define (initial-lrs o)
      (map (lambda (grp) (hash-ref grp "initial_lr"))
           (hash-ref (optimizer-state-dict o net) "param_groups")))
    (check-equal? (initial-lrs opt) '(1/250))
    (define rebuilt-opt (sgd (parameters net) #:lr 0.7))
    (define rebuilt (cycle rebuilt-opt))
    (load-optimizer-state! rebuilt-opt net (optimizer-state-dict opt net))
    (load-scheduler-state! rebuilt state)
    (check-equal? (hash-ref (scheduler-state-dict rebuilt) "base_lrs") '(1/250))
    (check-equal? (initial-lrs rebuilt-opt) '(1/250))))
