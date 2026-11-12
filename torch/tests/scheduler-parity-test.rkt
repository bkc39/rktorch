#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           (only-in "../main.rkt" zeros)
           (only-in "../nn.rkt"
                    Parameter chained-scheduler cosine-annealing-lr cosine-annealing-warm-restarts
                    exponential-lr lambda-lr linear-lr multi-step-lr one-cycle-lr param-group
                    scheduler-rate sequential-lr sgd step! step-lr)
           (only-in (submod "../nn/scheduler.rkt" private) closed-form-rates)
           (only-in "private/python-env.rkt" python-check python-torch-available?))

  (define (two-group-opt)
    (sgd (list (param-group (list (Parameter (zeros 1))) #:lr 0.1)
               (param-group (list (Parameter (zeros 1))) #:lr 0.02))
         #:lr 0.1))

  (define cases
    `((step ,(lambda (o) (step-lr o #:step-size 3 #:gamma 0.5)))
      (multi_step ,(lambda (o) (multi-step-lr o #:milestones '(2 5 9) #:gamma 0.1)))
      (exponential ,(lambda (o) (exponential-lr o #:gamma 0.9)))
      (cosine ,(lambda (o) (cosine-annealing-lr o #:t-max 10 #:eta-min 0.01)))
      (linear ,(lambda (o) (linear-lr o #:start-factor 0.25 #:end-factor 1.0 #:total-iters 4)))
      (one_cycle ,(lambda (o) (one-cycle-lr o #:max-lr 1.0 #:total-steps 12)))
      (lambda ,(lambda (o) (lambda-lr o (lambda (t) (/ 1.0 (add1 t))))))
      (chained_exponential_step
       ,(lambda (o) (chained-scheduler (list (exponential-lr o #:gamma 0.9)
                                             (step-lr o #:step-size 3 #:gamma 0.5)))))
      (chained_linear_cosine
       ,(lambda (o) (chained-scheduler
                     (list (linear-lr o #:start-factor 0.25 #:end-factor 1.0 #:total-iters 4)
                           (cosine-annealing-lr o #:t-max 10 #:eta-min 0.01)))))
      (sequential_linear_cosine
       ,(lambda (o) (sequential-lr
                     o (list (linear-lr o #:start-factor 0.25 #:end-factor 1.0 #:total-iters 4)
                             (cosine-annealing-lr o #:t-max 8 #:eta-min 0.01))
                     #:milestones '(4))))
      (sequential_exponential_step
       ,(lambda (o) (sequential-lr o (list (exponential-lr o #:gamma 0.9)
                                           (step-lr o #:step-size 2 #:gamma 0.5))
                                   #:milestones '(3))))
      (sequential_exponential_warm_restarts
       ,(lambda (o) (sequential-lr
                     o (list (exponential-lr o #:gamma 0.9)
                             (cosine-annealing-warm-restarts o #:t-0 2 #:eta-min 0.01))
                     #:milestones '(3))))
      (sequential_chained_linear
       ,(lambda (o) (sequential-lr
                     o (list (chained-scheduler (list (exponential-lr o #:gamma 0.9)
                                                      (step-lr o #:step-size 3 #:gamma 0.5)))
                             (linear-lr o #:start-factor 0.5 #:end-factor 1.0 #:total-iters 3))
                     #:milestones '(5))))
      (sequential_step_lambda
       ,(lambda (o) (sequential-lr o (list (step-lr o #:step-size 2 #:gamma 0.5)
                                           (lambda-lr o (lambda (t) (/ 1.0 (add1 t)))))
                                   #:milestones '(4))))
      (warm_restarts_3_2
       ,(lambda (o) (cosine-annealing-warm-restarts o #:t-0 3 #:t-mult 2 #:eta-min 0.01)))
      (warm_restarts_4_1
       ,(lambda (o) (cosine-annealing-warm-restarts o #:t-0 4 #:t-mult 1 #:eta-min 0)))))

  (define (stepped make read)
    (define s (make (two-group-opt)))
    (cons (read s) (for/list ([_ (in-range 12)]) (step! s) (read s))))

  (define (rates make)
    (stepped make (lambda (s) (list (scheduler-rate s #:group 0) (scheduler-rate s #:group 1)))))

  (cond
    [(not (python-torch-available?))
     (displayln "[scheduler-parity-test] skipped: python3 `torch` not available")]
    [else
     (define j (python-check "scheduler_chains.py"))
     (for ([c (in-list cases)])
       (match-define (list name make) c)
       (define want (hash-ref j name))
       (define got (rates make))
       (check-equal? (length got) (length want) (format "G2-L1: ~a: step count" name))
       (for ([g (in-list got)] [w (in-list want)] [t (in-naturals)])
         (check-equal? g w (format "G2-L1: ~a: get_last_lr() at step ~a" name t))))
     (for ([(name want) (in-hash (hash-ref j 'groups))])
       (match-define (list _ make) (assq name cases))
       (for ([g (in-list (stepped make closed-form-rates))] [w (in-list want)] [t (in-naturals)])
         (check-equal? g w (format "G1-L10: ~a: _get_closed_form_lr() at step ~a" name t))))]))
