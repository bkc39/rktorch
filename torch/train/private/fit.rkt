#lang racket/base

(require (only-in racket/match match match*)
         (only-in racket/math exact-floor)
         (only-in (submod "../../data/loader.rkt" private) with-traversal)
         (only-in "../../foreign.rkt" / backward! detach to)
         (only-in "../../foreign/autograd-ops.rkt" reset-grads!)
         (only-in "../../nn/layer.rkt" layer? parameters)
         (only-in "../../nn/optim.rkt" step! zero-grads!)
         (only-in "../logging.rkt" logger-log-hyperparams!)
         (only-in "dispatch.rkt"
                  call-callbacks call-hooks call-trainer call-trainer/values finish!
                  install-callbacks! note-batch-size! publish-epoch! publish-steps!
                  set-up! tear-down! transfer with-failure-protocol)
         (only-in "evaluate.rkt" evaluate)
         (only-in "state.rkt"
                  run-callbacks run-device run-epoch run-global-step
                  run-loggers run-optimizers run-schedulers run-skip-epoch?
                  run-skip-window? run-stepped? run-stop? run-trainer set-run-batch-index!
                  set-run-epoch! set-run-global-step! set-run-loggers!
                  set-run-optimizers! set-run-schedulers! set-run-skip-epoch?!
                  set-run-skip-window?! set-run-stage! set-run-step-pending?!
                  set-run-stepped?!)
         (only-in "types.rkt"
                  scheduler-config-frequency scheduler-config-interval
                  scheduler-config-scheduler trainer-hook))

(provide (struct-out plan)
         run-fit
         start-entry!)

(struct plan (move-to val max-epochs max-steps accumulate val-every val-interval
              sanity train-count val-count log-every fast?))

(define (start-entry! r who callbacks loggers move-to #:fast? [fast? #f])
  (set-run-loggers! r loggers)
  (install-callbacks! r who callbacks #:fast? fast?)
  (set-up! r who)
  (match move-to
    [#f (void)]
    [dev (for ([layer (in-list (cons (run-trainer r) (run-callbacks r)))]
               #:when (layer? layer))
           (to layer dev))]))

(define (step-schedulers! r interval index)
  (for ([c (in-list (run-schedulers r))]
        #:when (and (eq? (scheduler-config-interval c) interval)
                    (zero? (modulo (add1 index) (scheduler-config-frequency c)))))
    (step! (scheduler-config-scheduler c))))

(define (done? r p)
  (or (run-stop? r)
      (and (plan-max-epochs p) (>= (run-epoch r) (plan-max-epochs p)))
      (and (plan-max-steps p) (>= (run-global-step r) (plan-max-steps p)))))

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
  (evaluate r (plan-val p) count stage 'validation))

(define (backward-pass! r scaled)
  (match* ((run-skip-window? r) scaled)
    [(#f (? values))
     (call-hooks r 'on-before-backward scaled)
     (match (run-skip-window? r)
       [#f (backward! scaled)
           (call-hooks r 'on-after-backward)]
       [_ (void)])]
    [(_ _) (set-run-skip-window?! r #t)]))

(define (optimizer-step! r)
  (for ([opt (in-list (run-optimizers r))])
    (call-hooks r 'on-before-optimizer-step opt)
    (step! opt))
  (set-run-global-step! r (add1 (run-global-step r)))
  (set-run-stepped?! r #t))

(define (train-batch! r p i raw)
  (define a (plan-accumulate p))
  (define last? (= i (sub1 (plan-train-count p))))
  (define window-end? (or last? (zero? (modulo (add1 i) a))))
  (set-run-batch-index! r i)
  (set-run-stepped?! r #f)
  (set-run-step-pending?! r #t)
  (when (zero? (modulo i a)) (set-run-skip-window?! r #f))
  (define batch (transfer r (run-device r) raw))
  (note-batch-size! r batch)
  (apply call-hooks r 'on-train-batch-start batch)
  (match (run-skip-epoch? r)
    [#t
     (set-run-step-pending?! r #f)
     (step-schedulers! r 'epoch (run-epoch r))
     #f]
    [#f
     (define-values (loss extra) (apply values (call-trainer/values r 'training-step batch)))
     (when (zero? (modulo i a))
       (for ([opt (in-list (run-optimizers r))])
         (call-hooks r 'on-before-zero-grad opt)
         (zero-grads! opt)))
     (define scaled (and loss (if (= a 1) loss (/ loss a))))
     (backward-pass! r scaled)
     (when (and window-end? (not (run-skip-window? r)))
       (optimizer-step! r)
       (step-schedulers! r 'step i))
     (set-run-step-pending?! r #f)
     (when last? (step-schedulers! r 'epoch (run-epoch r)))
     (apply call-hooks r 'on-train-batch-end (and scaled (detach scaled)) extra batch)
     (when (and (run-stepped? r) (zero? (modulo (run-global-step r) (plan-log-every p))))
       (publish-steps! r))
     (when (val-due? r p i)
       (when window-end? (reset-grads! (parameters (run-trainer r))))
       (validate! r p 'validate (plan-val-count p)))
     (not (or (run-stop? r)
              (and (plan-max-steps p) (>= (run-global-step r) (plan-max-steps p)))))]))

(define (train-epoch! r p train)
  (set-run-stage! r 'train)
  (call-hooks r 'on-train-epoch-start)
  (with-traversal ([_ batch] train)
    (let next ([i 0])
      (cond
        [(>= i (plan-train-count p)) (void)]
        [(train-batch! r p i (call-with-values (lambda () (batch i)) list)) (next (add1 i))]
        [else (void)])))
  (set-run-batch-index! r #f)
  (set-run-skip-epoch?! r #f)
  (call-hooks r 'on-train-epoch-end #:between (lambda () (publish-epoch! r 'train)))
  (publish-epoch! r 'train)
  (set-run-epoch! r (add1 (run-epoch r))))

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

(define (run-fit r p train callbacks loggers)
  (with-failure-protocol r 'fit
    (set-run-stage! r 'train)
    (start-entry! r 'fit callbacks loggers (plan-move-to p) #:fast? (plan-fast? p))
    (configure-optimizers! r)
    (log-hyperparameters! r)
    (call-hooks r 'on-fit-start)
    (when (and (plan-val p) (positive? (plan-sanity p)))
      (call-callbacks r 'on-sanity-check-start (run-callbacks r))
      (validate! r p 'sanity-check (min (plan-sanity p) (plan-val-count p)))
      (call-callbacks r 'on-sanity-check-end (run-callbacks r)))
    (call-hooks r 'on-train-start)
    (let loop ()
      (unless (done? r p)
        (train-epoch! r p train)
        (loop)))
    (call-hooks r 'on-train-end)
    (call-hooks r 'on-fit-end)
    (tear-down! r 'fit)
    (finish! r 'success))
  r)
