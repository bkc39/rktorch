#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i </c <=/c >/c >=/c and/c any/c contract-out listof
                  non-empty-listof real-in unsupplied-arg?)
         (only-in racket/flonum ->fl flexpt)
         (only-in racket/generic define/generic)
         (only-in racket/list count)
         (only-in racket/match match match-define struct*)
         (only-in racket/math pi)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "optim-groups.rkt"
                  store-initial-lrs! store-set-each! store-update-each! store-values)
         (only-in "optim.rkt"
                  gen:optimizer group-index? optimizer-parameters optimizer-root optimizer-store
                  optimizer?))

(provide (contract-out [scheduler? (-> any/c boolean?)]))

(module+ private
  (provide closed-form-rates))

(struct scheduler (optimizer rule [last #:mutable] [last-rates #:mutable])
  #:constructor-name make-scheduler
  #:name scheduler-value ;; noqa
  #:methods gen:optimizer
  [(define/generic inner-parameters optimizer-parameters)
   (define/generic inner-store optimizer-store)
   (define/generic inner-root optimizer-root)
   (define (optimizer-parameters s) (inner-parameters (scheduler-optimizer s)))
   (define (optimizer-store s) (inner-store (scheduler-optimizer s)))
   (define (optimizer-root s) (inner-root (scheduler-optimizer s)))
   (define (optimizer-step! s) (advance! s))])

(struct shape (bases next closed))

(struct restarts (bases t-0 t-mult eta-min [t-cur #:mutable] [t-i #:mutable]))

(struct chain (schedulers))

(struct sequence (schedulers milestones))

(define (current-lrs s)
  (store-values (optimizer-store s) 'lr))

(define (settle! s t rates)
  (set-scheduler-last! s t)
  (set-scheduler-last-rates! s rates))

(define (write! s t f xs)
  (settle! s t (store-update-each! (optimizer-store s) 'lr f xs)))

(define (absolute rate _lr)
  rate)

(define (write-rates! s t rates)
  (write! s t absolute rates))

(define (write-shape! s bases next t initial?)
  (write! s t (lambda (base lr) (next base lr t initial?)) bases))

(define (restart-rate eta-min base cur period)
  (+ eta-min (/ (* (- base eta-min) (+ 1.0 (cos (/ (* pi (->fl cur)) (->fl period))))) 2.0)))

(define (restart-rates bases eta-min cur period)
  (map (lambda (base) (restart-rate eta-min base cur period)) bases))

(define (restart-step! s r)
  (match-define (restarts bases t-0 t-mult eta-min t-cur t-i) r)
  (define last (scheduler-last s))
  (define-values (epoch cur period)
    (cond
      [(negative? last) (values 0 0 t-0)]
      [(>= (add1 t-cur) t-i) (values (add1 last) (modulo (add1 t-cur) t-i) (* t-i t-mult))]
      [else (values (add1 last) (add1 t-cur) t-i)]))
  (define rates (restart-rates bases eta-min cur period))
  (set-restarts-t-cur! r cur)
  (set-restarts-t-i! r period)
  (write-rates! s epoch rates))

(define (milestones-reached milestones t)
  (count (lambda (m) (<= m t)) milestones))

(define (bisect-right milestones t)
  (let search ([lo 0] [hi (vector-length milestones)])
    (define mid (quotient (+ lo hi) 2))
    (cond
      [(>= lo hi) lo]
      [(< t (vector-ref milestones mid)) (search lo mid)]
      [else (search (add1 mid) hi)])))

(define (sequence-step! s subs milestones)
  (define t (add1 (scheduler-last s)))
  (define i (bisect-right milestones t))
  (define sub (vector-ref subs i))
  (define crossed? (and (positive? i) (= (vector-ref milestones (sub1 i)) t)))
  (if crossed? (start-at-zero! sub) (advance! sub))
  (settle! s t (scheduler-last-rates sub)))

(define (advance! s)
  (define t (add1 (scheduler-last s)))
  (match (scheduler-rule s)
    [(struct* shape ([bases bases] [next next])) (write-shape! s bases next t #f)]
    [(and (struct* restarts ()) r) (restart-step! s r)]
    [(struct* chain ([schedulers subs]))
     (for-each advance! subs)
     (settle! s t (current-lrs s))]
    [(struct* sequence ([schedulers subs] [milestones milestones]))
     (sequence-step! s subs milestones)]))

(define (initial-step! s)
  (match (scheduler-rule s)
    [(struct* shape ([bases bases] [next next]))
     (write-shape! s bases next (add1 (scheduler-last s)) #t)]
    [_ (advance! s)]))

(define (start-at-zero! s)
  (match (scheduler-rule s)
    [(struct* shape ([bases bases] [closed (? procedure? closed)]))
     (write-rates! s 0 (map (lambda (b) (closed b 0)) bases))]
    [(struct* shape ([bases bases] [next next])) (write-shape! s bases next 0 #f)]
    [(struct* restarts ([bases bases] [eta-min eta-min] [t-cur t-cur] [t-i t-i]))
     (write-rates! s 0 (restart-rates bases eta-min t-cur t-i))]
    [_ (raise-arguments-error 'sequential-lr
                              "a chained or sequential schedule cannot start at a milestone"
                              "schedule" s)]))

(define (step-back! s)
  (set-scheduler-last! s (sub1 (scheduler-last s))))

(define (undo! s)
  (match (scheduler-rule s)
    [(struct* chain ([schedulers subs]))
     (for-each undo! subs)
     (step-back! s)]
    [(struct* sequence ([schedulers subs])) (for ([sub (in-vector subs)]) (undo! sub))]
    [_ (step-back! s)]))

(define (build opt rule-of #:initial [initial #f])
  (define bases (store-initial-lrs! (optimizer-store opt) initial))
  (define s (make-scheduler opt (rule-of bases) -1 #f))
  (initial-step! s)
  s)

(define (shaped opt next #:closed [closed #f] #:initial [initial #f])
  (build opt (lambda (bases) (shape bases next closed)) #:initial initial))

(define (closed-form-rates s)
  (match (scheduler-rule s)
    [(struct* shape ([bases bases] [closed (? procedure? closed)]))
     (map (lambda (b) (real->double-flonum (closed b (scheduler-last s)))) bases)]
    [_ (scheduler-last-rates s)]))

(define/contract-out (scheduler-step-count s) ;; noqa
  (-> scheduler? exact-nonnegative-integer?)
  (max 0 (scheduler-last s)))

(define/contract-out (scheduler-rate s #:group [group 0]) ;; noqa
  (->i ([s scheduler?]) (#:group [group exact-nonnegative-integer?])
       #:pre/name (s group) "the optimizer has no parameter group with this index"
       (or (unsupplied-arg? group) (group-index? s group))
       [_ real?])
  (list-ref (scheduler-last-rates s) group))

(define/contract-out (step-lr opt #:step-size step-size #:gamma [gamma 0.1]) ;; noqa
  (->* [optimizer? #:step-size exact-positive-integer?] [#:gamma real?]
       scheduler?)
  (let ([gamma (real->double-flonum gamma)])
    (shaped opt
            (lambda (_base lr t _initial?)
              (if (or (zero? t) (positive? (modulo t step-size))) lr (* lr gamma)))
            #:closed (lambda (base t) (* base (flexpt gamma (->fl (quotient t step-size))))))))

(define/contract-out (multi-step-lr opt ;; noqa
                                    #:milestones milestones
                                    #:gamma [gamma 0.1])
  (->* [optimizer? #:milestones (listof exact-nonnegative-integer?)]
       [#:gamma real?]
       scheduler?)
  (let ([gamma (real->double-flonum gamma)])
    (shaped opt
            (lambda (_base lr t _initial?)
              (match (count (lambda (m) (= m t)) milestones)
                [0 lr]
                [n (* lr (flexpt gamma (->fl n)))]))
            #:closed (lambda (base t)
                       (* base (flexpt gamma (->fl (milestones-reached milestones t))))))))

(define/contract-out (exponential-lr opt #:gamma gamma) ;; noqa
  (-> optimizer? #:gamma real? scheduler?)
  (let ([gamma (real->double-flonum gamma)])
    (shaped opt
            (lambda (_base lr _t initial?) (if initial? lr (* lr gamma)))
            #:closed (lambda (base t) (* base (flexpt gamma (->fl t)))))))

(define/contract-out (cosine-annealing-lr opt ;; noqa
                                          #:t-max t-max
                                          #:eta-min [eta-min 0.0])
  (->* [optimizer? #:t-max exact-positive-integer?] [#:eta-min real?]
       scheduler?)
  (define (one-plus-cos t) (+ 1.0 (cos (/ (* pi (->fl t)) (->fl t-max)))))
  (let ([eta-min (real->double-flonum eta-min)])
    (shaped opt
            (lambda (base lr t initial?)
              (cond
                [initial? lr]
                [(zero? (modulo (- t 1 t-max) (* 2 t-max)))
                 (+ lr (/ (* (- base eta-min) (- 1.0 (cos (/ pi (->fl t-max))))) 2.0))]
                [else (+ (* (/ (one-plus-cos t) (one-plus-cos (sub1 t))) (- lr eta-min)) eta-min)]))
            #:closed (lambda (base t) (+ eta-min (/ (* (- base eta-min) (one-plus-cos t)) 2.0))))))

(define/contract-out (linear-lr opt ;; noqa
                                #:start-factor [start-factor (/ 1.0 3.0)]
                                #:end-factor [end-factor 1.0]
                                #:total-iters [total-iters 5])
  (->* [optimizer?]
       [#:start-factor (and/c (>/c 0) (<=/c 1)) #:end-factor (real-in 0 1)
        #:total-iters exact-positive-integer?]
       scheduler?)
  (define start (real->double-flonum start-factor))
  (define rise (- (real->double-flonum end-factor) start))
  (define total (->fl total-iters))
  (shaped opt
         (lambda (_base lr t initial?)
           (cond
             [(zero? t) (* lr start)]
             [(or initial? (> t total-iters)) lr]
             [else (* lr (+ 1.0 (/ rise (+ (* total start) (* (->fl (sub1 t)) rise)))))]))
         #:closed (lambda (base t)
                    (* base (+ start (/ (* rise (->fl (min total-iters t))) total))))))

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
  (shaped opt
         #:initial initial
         (lambda (_base _lr t _initial?)
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
  (shaped opt (lambda (base _lr t _initial?) (* base (real->double-flonum (factor t))))))

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
  (match-define (cons first _) schedulers)
  (make-scheduler (scheduler-optimizer first) (chain schedulers) 0 (current-lrs first)))

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
  (define st (optimizer-store opt))
  (store-set-each! st 'lr (store-initial-lrs! st #f))
  (for-each undo! schedulers)
  (match-define (and rule (sequence subs marks))
    (sequence (list->vector schedulers) (list->vector milestones)))
  (define first (vector-ref subs (bisect-right marks 0)))
  (initial-step! first)
  (make-scheduler opt rule 0 (scheduler-last-rates first)))

(define/contract-out (cosine-annealing-warm-restarts opt ;; noqa
                                                     #:t-0 t-0
                                                     #:t-mult [t-mult 1]
                                                     #:eta-min [eta-min 0.0])
  (->* (optimizer? #:t-0 exact-positive-integer?)
       (#:t-mult exact-positive-integer? #:eta-min real?)
       scheduler?)
  (build opt (lambda (bases) (restarts bases t-0 t-mult (real->double-flonum eta-min) -1 t-0))))
