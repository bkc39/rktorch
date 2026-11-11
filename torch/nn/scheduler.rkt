#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i </c <=/c >/c >=/c and/c any/c contract-out listof
                  non-empty-listof real-in unsupplied-arg?)
         (only-in racket/generic define/generic)
         (only-in racket/math pi)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "optim.rkt"
                  gen:optimizer optimizer-lr optimizer-parameters optimizer-root
                  optimizer-set-lr! optimizer? param-group-count))

(provide (contract-out [scheduler? (-> any/c boolean?)]))

(struct scheduler (optimizer base-lr lr-at [last #:mutable]
                             [last-rate #:mutable])
  #:constructor-name make-scheduler
  #:name scheduler-value ;; noqa
  #:methods gen:optimizer
  [(define/generic inner-parameters optimizer-parameters)
   (define/generic inner-lr optimizer-lr)
   (define/generic inner-set-lr! optimizer-set-lr!)
   (define/generic inner-root optimizer-root)
   (define (optimizer-parameters s) (inner-parameters (scheduler-optimizer s)))
   (define (optimizer-lr s) (inner-lr (scheduler-optimizer s)))
   (define (optimizer-set-lr! s lr) (inner-set-lr! (scheduler-optimizer s) lr))
   (define (optimizer-root s) (inner-root (scheduler-optimizer s)))
   (define (optimizer-step! s)
     (apply-rate! s (add1 (scheduler-last s))))])

(define (apply-rate! s t)
  (define lr ((scheduler-lr-at s) (scheduler-base-lr s) t))
  (set-scheduler-last! s t)
  (set-scheduler-last-rate! s lr)
  (optimizer-set-lr! (scheduler-optimizer s) lr))

(define (build opt lr-at)
  (define s (make-scheduler opt (optimizer-lr opt) lr-at 0 #f))
  (apply-rate! s 0)
  s)

(define/contract-out (scheduler-step-count s) ;; noqa
  (-> scheduler? exact-nonnegative-integer?)
  (scheduler-last s))

(define/contract-out (scheduler-rate s #:group [group 0]) ;; noqa
  (->i ([s scheduler?]) (#:group [group exact-nonnegative-integer?])
       #:pre/name (s group) "the optimizer has no parameter group with this index"
       (or (unsupplied-arg? group) (zero? group) (< group (param-group-count s)))
       [_ real?])
  (cond
    [(zero? group) (scheduler-last-rate s)]
    [else (error 'unimplemented)]))

(define/contract-out (step-lr opt #:step-size step-size #:gamma [gamma 0.1]) ;; noqa
  (->* [optimizer? #:step-size exact-positive-integer?] [#:gamma real?]
       scheduler?)
  (build opt (lambda (base t) (* base (expt gamma (quotient t step-size))))))

(define/contract-out (multi-step-lr opt ;; noqa
                                    #:milestones milestones
                                    #:gamma [gamma 0.1])
  (->* [optimizer? #:milestones (listof exact-nonnegative-integer?)]
       [#:gamma real?]
       scheduler?)
  (build opt
         (lambda (base t)
           (* base (expt gamma (for/sum ([m (in-list milestones)])
                                 (if (<= m t) 1 0)))))))

(define/contract-out (exponential-lr opt #:gamma gamma) ;; noqa
  (-> optimizer? #:gamma real? scheduler?)
  (build opt (lambda (base t) (* base (expt gamma t)))))

(define/contract-out (cosine-annealing-lr opt ;; noqa
                                          #:t-max t-max
                                          #:eta-min [eta-min 0.0])
  (->* [optimizer? #:t-max exact-positive-integer?] [#:eta-min real?]
       scheduler?)
  (build opt
         (lambda (base t)
           (+ eta-min
              (* (- base eta-min)
                 (/ (+ 1.0 (cos (/ (* pi t) t-max))) 2.0))))))

(define/contract-out (linear-lr opt ;; noqa
                                #:start-factor [start-factor (/ 1.0 3.0)]
                                #:end-factor [end-factor 1.0]
                                #:total-iters [total-iters 5])
  (->* [optimizer?]
       [#:start-factor (and/c (>/c 0) (<=/c 1)) #:end-factor (real-in 0 1)
        #:total-iters exact-positive-integer?]
       scheduler?)
  (build opt
         (lambda (base t)
           (* base
              (+ start-factor
                 (* (- end-factor start-factor)
                    (/ (min t total-iters) total-iters)))))))

;; the two cosine phases of torch's OneCycleLR (anneal_strategy "cos",
;; three_phase False); momentum is left alone, cycle_momentum=False there
(define/contract-out (one-cycle-lr opt ;; noqa
                                   #:max-lr max-lr
                                   #:total-steps total-steps
                                   #:pct-start [pct-start 0.3]
                                   #:div-factor [div-factor 25.0]
                                   #:final-div-factor [final-div-factor 1e4])
  (->* [optimizer? #:max-lr (>/c 0) #:total-steps exact-positive-integer?]
       [#:pct-start (and/c (>=/c 0) (</c 1)) #:div-factor (>/c 0)
        #:final-div-factor (>/c 0)]
       scheduler?)
  (define initial (/ max-lr div-factor))
  (define final (/ initial final-div-factor))
  (define up-end (- (* pct-start total-steps) 1.0))
  (define down-end (- total-steps 1.0))
  (define (anneal start end pct)
    (+ end (* (/ (- start end) 2.0) (+ 1.0 (cos (* pi pct))))))
  (build opt
         (lambda (_base t)
           (when (> t total-steps)
             (raise-arguments-error 'one-cycle-lr
                                    "stepped past the cycle"
                                    "total-steps" total-steps
                                    "step" t))
           (if (<= t up-end)
               (anneal initial max-lr (/ t up-end))
               (anneal max-lr final (/ (- t up-end) (- down-end up-end)))))))

(define/contract-out (lambda-lr opt factor) ;; noqa
  (-> optimizer? (-> exact-nonnegative-integer? real?) scheduler?)
  (build opt (lambda (base t) (* base (factor t)))))

(define/contract-out (scheduler-optimizer-of s) ;; noqa
  (-> scheduler? any/c)
  (scheduler-optimizer s))

(define (wraps? s opt)
  (eq? (optimizer-root s) (optimizer-root opt)))

(define/contract-out (chained-scheduler schedulers) ;; noqa
  (->i ([schedulers (non-empty-listof scheduler?)])
       #:pre/name (schedulers) "every schedule in a chain wraps the same optimizer"
       (andmap (lambda (s) (wraps? s (car schedulers))) schedulers)
       [_ scheduler?])
  (error 'unimplemented))

(define/contract-out (sequential-lr opt schedulers #:milestones milestones) ;; noqa
  (->i ([opt optimizer?] [schedulers (non-empty-listof scheduler?)]
        #:milestones [milestones (listof exact-nonnegative-integer?)])
       #:pre/name (opt) "sequential-lr takes an optimizer, not a schedule"
       (not (scheduler? opt))
       #:pre/name (opt schedulers) "every schedule wraps opt"
       (andmap (lambda (s) (wraps? s opt)) schedulers)
       #:pre/name (schedulers milestones) "sequential-lr takes one milestone fewer than schedules"
       (= (length milestones) (sub1 (length schedulers)))
       [_ scheduler?])
  (error 'unimplemented))

(define/contract-out (cosine-annealing-warm-restarts opt ;; noqa
                                                     #:t-0 t-0
                                                     #:t-mult [t-mult 1]
                                                     #:eta-min [eta-min 0.0])
  (->* (optimizer? #:t-0 exact-positive-integer?)
       (#:t-mult exact-positive-integer? #:eta-min real?)
       scheduler?)
  (error 'unimplemented))
