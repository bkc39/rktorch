#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           (only-in "../nn.rkt"
                    Linear chained-scheduler cosine-annealing-lr cosine-annealing-warm-restarts
                    exponential-lr lambda-lr linear-lr multi-step-lr one-cycle-lr parameters
                    scheduler-state-dict sequential-lr sgd step! step-lr)
           (only-in "private/optim-oracle.rkt" call-with-optim-oracle oracle-json))

  (define kinds
    `(("step" ,(lambda (o) (step-lr o #:step-size 2 #:gamma 0.5)))
      ("multi-step" ,(lambda (o) (multi-step-lr o #:milestones '(2 4) #:gamma 0.5)))
      ("exponential" ,(lambda (o) (exponential-lr o #:gamma 0.9)))
      ("cosine" ,(lambda (o) (cosine-annealing-lr o #:t-max 5 #:eta-min 0.001)))
      ("linear" ,(lambda (o) (linear-lr o #:start-factor 0.5 #:total-iters 4)))
      ("one-cycle" ,(lambda (o) (one-cycle-lr o #:max-lr 0.1 #:total-steps 10)))
      ("lambda" ,(lambda (o) (lambda-lr o (lambda (t) (expt 0.95 t)))))
      ("chained" ,(lambda (o) (chained-scheduler (list (step-lr o #:step-size 2 #:gamma 0.5)
                                                       (exponential-lr o #:gamma 0.9)))))
      ("sequential" ,(lambda (o) (sequential-lr o (list (linear-lr o #:start-factor 0.5
                                                                   #:total-iters 2)
                                                        (cosine-annealing-lr o #:t-max 5))
                                                #:milestones '(2))))
      ("warm-restarts" ,(lambda (o) (cosine-annealing-warm-restarts o #:t-0 3 #:t-mult 2)))))

  (define (keys-with-a-step-count torch)
    (define keys (map symbol->string (hash-keys torch)))
    (sort (if (member "last_epoch" keys) keys (cons "last_epoch" keys)) string<?))

  (define (check-state label ours torch)
    (check-equal? (sort (hash-keys ours) string<?) (keys-with-a-step-count torch)
                  (format "~a: the keys" label))
    (for ([(key v) (in-hash torch)] #:when (hash-has-key? ours (symbol->string key)))
      (define name (symbol->string key))
      (define mine (hash-ref ours name))
      (define where (format "~a: ~a" label name))
      (when (list? v)
        (check-equal? (length mine) (length v) (format "~a's length" where)))
      (cond
        [(equal? name "last_epoch") (check-equal? mine v where)]
        [(equal? name "_schedulers")
         (for ([o (in-list mine)] [t (in-list v)] [i (in-naturals)])
           (check-state (format "~a[~a]" where i) o t))]
        [(list? v) (for ([a (in-list mine)] [b (in-list v)]) (check-= a b 1e-12 where))]
        [else (check-= mine v 1e-12 where)])))

  (call-with-optim-oracle
   'scheduler-state-parity-test
   (lambda (dir)
     (test-case "T3-L30: scheduler state after each step equals torch's state_dict"
       (for ([kind (in-list kinds)])
         (match-define (list name make) kind)
         (define opt (sgd (parameters (Linear 3 2)) #:lr 0.1 #:momentum 0.9))
         (define s (make opt))
         (for ([torch (in-list (hash-ref (oracle-json dir name) 'states))] [t (in-naturals)])
           (unless (zero? t) (step! s))
           (check-state (format "~a, step ~a" name t) (scheduler-state-dict s) torch)))))))
