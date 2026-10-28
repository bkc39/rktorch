#lang racket/base

(require (only-in racket/contract/base unsupplied-arg?)
         (only-in racket/match match match-define)
         (only-in racket/math exact-floor)
         (only-in (submod "../../data/loader.rkt" private) with-traversal)
         (only-in "../../data/loader.rkt" dataloader-length)
         (only-in "../../foreign.rkt" / backward! detach to)
         (only-in "../../foreign/autograd-ops.rkt" reset-grads!)
         (only-in "../../nn/layer.rkt" layer? parameters)
         (only-in "../../nn/optim.rkt" step! zero-grads!)
         (only-in "../logging.rkt" logger-log-hyperparams!)
         (only-in "book.rkt" epoch-end-reached!)
         (only-in "dispatch.rkt"
                  call-callbacks call-hooks call-trainer/values finish! install-callbacks!
                  note-batch-size! publish-epoch! publish-steps! set-up! tear-down! transfer
                  with-failure-protocol)
         (only-in "evaluate.rkt" evaluate)
         (only-in "state.rkt"
                  run-book run-callbacks run-device run-epoch run-global-step run-loggers
                  run-optimizers run-schedulers run-skip-epoch? run-skip-window? run-stepped?
                  run-stop? run-trainer set-run-batch-index! set-run-current-stage!
                  set-run-epoch! set-run-global-step! set-run-loggers! set-run-optimizers!
                  set-run-schedulers! set-run-skip-epoch?! set-run-skip-window?!
                  set-run-step-pending?! set-run-stepped?!)
         (only-in "types.rkt"
                  scheduler-config-frequency scheduler-config-interval
                  scheduler-config-scheduler trainer-hook))

(provide fit-plan
         given
         limited-batches
         run-fit
         start-entry!)

(define (given v [default #f])
  (if (unsupplied-arg? v) default v))

(define (limited-batches loader limit)
  (define n (dataloader-length loader))
  (match (given limit 1.0)
    [(? exact-integer? k) (min k n)]
    [share (exact-floor (* n share))]))

(struct plan (device train val callbacks loggers max-epochs max-steps accumulate
              val-every val-interval sanity train-count val-count log-every fast?))

(define (fit-plan train
                  #:val val
                  #:device device
                  #:callbacks callbacks
                  #:loggers loggers
                  #:max-epochs max-epochs
                  #:max-steps max-steps
                  #:accumulate accumulate
                  #:val-every val-every
                  #:val-interval val-interval
                  #:sanity sanity
                  #:limit-train limit-train
                  #:limit-val limit-val
                  #:log-every log-every
                  #:fast-dev-run fast-dev-run)
  (define fast (match fast-dev-run [#f #f] [#t 1] [k k]))
  (define (unless-fast v fast-v) (if fast fast-v v))
  (plan device train val callbacks
        (unless-fast loggers '())
        (unless-fast max-epochs 1)
        (or fast max-steps)
        accumulate
        (unless-fast val-every 1)
        (unless-fast val-interval #f)
        (unless-fast sanity 0)
        (limited-batches train (or fast limit-train))
        (and val (limited-batches val (or fast limit-val)))
        log-every
        (and fast #t)))

(define (start-entry! r who callbacks loggers move-to #:fast? [fast? #f])
  (set-run-loggers! r loggers)
  (install-callbacks! r who callbacks #:fast? fast?)
  (set-up! r who)
  (when move-to
    (for ([layer (in-list (cons (run-trainer r) (run-callbacks r)))]
          #:when (layer? layer))
      (to layer move-to))))

(define (step-schedulers! r interval index)
  (for ([c (in-list (run-schedulers r))]
        #:when (and (eq? (scheduler-config-interval c) interval)
                    (zero? (modulo (add1 index) (scheduler-config-frequency c)))))
    (step! (scheduler-config-scheduler c))))

(define (max-steps-reached? r p)
  (and (plan-max-steps p) (>= (run-global-step r) (plan-max-steps p))))

(define (done? r p)
  (or (run-stop? r)
      (and (plan-max-epochs p) (>= (run-epoch r) (plan-max-epochs p)))
      (max-steps-reached? r p)))

(define (epoch-over? r p)
  (or (run-stop? r) (run-skip-epoch? r) (max-steps-reached? r p)))

(define (window-start? p i)
  (zero? (modulo i (plan-accumulate p))))

(define (last-batch? p i)
  (= i (sub1 (plan-train-count p))))

(define (window-end? p i)
  (or (last-batch? p i) (zero? (modulo (add1 i) (plan-accumulate p)))))

(define (val-check-batches p)
  (match (plan-val-interval p)
    [#f (plan-train-count p)]
    [(? exact-integer? k) k]
    [share (max 1 (exact-floor (* share (plan-train-count p))))]))

(define (val-due? r p i)
  (and (plan-val p)
       (zero? (modulo (add1 (run-epoch r)) (plan-val-every p)))
       (or (run-stop? r) (zero? (modulo (add1 i) (val-check-batches p))))))

(define (validate! r p stage count)
  (evaluate r (plan-val p) count stage))

(define (backward-pass! r scaled)
  (cond
    [(not scaled) (set-run-skip-window?! r #t)]
    [(run-skip-window? r) (void)]
    [else
     (call-hooks r 'on-before-backward scaled)
     (unless (run-skip-window? r)
       (backward! scaled)
       (call-hooks r 'on-after-backward))]))

(define (optimizer-step! r i)
  (for ([opt (in-list (run-optimizers r))])
    (call-hooks r 'on-before-optimizer-step opt))
  (unless (run-skip-window? r)
    (for-each step! (run-optimizers r))
    (set-run-global-step! r (add1 (run-global-step r)))
    (set-run-stepped?! r #t)
    (step-schedulers! r 'step i)))

(define (train-step! r p i batch)
  (match-define (list loss extra) (call-trainer/values r 'training-step batch))
  (when (window-start? p i)
    (for ([opt (in-list (run-optimizers r))])
      (call-hooks r 'on-before-zero-grad opt)
      (zero-grads! opt)))
  (define a (plan-accumulate p))
  (define scaled (and loss (if (= a 1) loss (/ loss a))))
  (backward-pass! r scaled)
  (when (and (window-end? p i) (not (run-skip-window? r)))
    (optimizer-step! r i))
  (set-run-step-pending?! r #f)
  (when (last-batch? p i)
    (step-schedulers! r 'epoch (run-epoch r)))
  (apply call-hooks r 'on-train-batch-end (and scaled (detach scaled)) extra batch)
  (when (and (run-stepped? r) (zero? (modulo (run-global-step r) (plan-log-every p))))
    (publish-steps! r 'train (run-global-step r)))
  (define validate? (val-due? r p i))
  (when (and validate? (window-end? p i))
    (reset-grads! (parameters (run-trainer r))))
  (when validate?
    (validate! r p 'validate (plan-val-count p))))

(define (train-batch! r p i raw)
  (set-run-batch-index! r i)
  (set-run-stepped?! r #f)
  (set-run-step-pending?! r #t)
  (when (window-start? p i)
    (set-run-skip-window?! r #f))
  (define batch (transfer r (run-device r) raw))
  (note-batch-size! r batch)
  (apply call-hooks r 'on-train-batch-start batch)
  (cond
    [(run-skip-epoch? r)
     (set-run-step-pending?! r #f)
     (step-schedulers! r 'epoch (run-epoch r))]
    [else (train-step! r p i batch)]))

(define (check-progress! r p steps-before)
  (when (and (not (plan-max-epochs p))
             (not (run-stop? r))
             (= steps-before (run-global-step r)))
    (error 'fit
           (string-append "an epoch took no optimizer step, so #:max-steps ~a is never"
                          " reached; bound the fit with #:max-epochs")
           (plan-max-steps p))))

(define (train-epoch! r p)
  (define steps-before (run-global-step r))
  (set-run-current-stage! r 'train)
  (call-hooks r 'on-train-epoch-start)
  (with-traversal ([_ batch] (plan-train p))
    (for ([i (in-range (plan-train-count p))]
          #:break (epoch-over? r p))
      (train-batch! r p i (batch i))))
  (set-run-batch-index! r #f)
  (set-run-skip-epoch?! r #f)
  (epoch-end-reached! (run-book r) 'train)
  (call-hooks r 'on-train-epoch-end)
  (publish-epoch! r 'train)
  (set-run-epoch! r (add1 (run-epoch r)))
  (check-progress! r p steps-before))

(define (configure-optimizers! r)
  (define-values (optimizers schedulers)
    (match (call-trainer/values r 'configure-optimizers '())
      [#f (values '() '())]
      [(list optimizers schedulers) (values optimizers schedulers)]))
  (set-run-optimizers! r optimizers)
  (set-run-schedulers! r schedulers))

(define (log-hyperparameters! r)
  (define tr (run-trainer r))
  (define params ((trainer-hook tr 'hyperparameters) tr))
  (for ([l (in-list (run-loggers r))])
    (logger-log-hyperparams! l params)))

(define (run-fit r p)
  (with-failure-protocol r 'fit
    (set-run-current-stage! r 'train)
    (start-entry! r 'fit (plan-callbacks p) (plan-loggers p) (plan-device p)
                  #:fast? (plan-fast? p))
    (configure-optimizers! r)
    (log-hyperparameters! r)
    (call-hooks r 'on-fit-start)
    (when (and (plan-val p) (positive? (plan-sanity p)))
      (call-callbacks r 'on-sanity-check-start (run-callbacks r))
      (validate! r p 'sanity-check (min (plan-sanity p) (plan-val-count p)))
      (call-callbacks r 'on-sanity-check-end (run-callbacks r)))
    (call-hooks r 'on-train-start)
    (for ([_ (in-naturals)]
          #:break (done? r p))
      (train-epoch! r p))
    (call-hooks r 'on-train-end)
    (call-hooks r 'on-fit-end)
    (tear-down! r 'fit))
  (finish! r 'success)
  r)
