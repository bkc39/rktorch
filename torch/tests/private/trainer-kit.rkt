#lang racket/base

(require (for-syntax racket/base
                     (only-in syntax/parse/pre id)
                     (only-in "../../train/private/hooks.rkt"
                              callback-hook? definable-hooks hook-batch?
                              hook-formals hook-name trainer-hook?))
         (only-in racket/match match)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../../data/loader.rkt" dataloader tensor-dataset)
         (only-in "../../main.rkt" ones zeros)
         (only-in "../../nn.rkt"
                  Linear learning-rate mse-loss parameters sgd step-lr)
         (only-in "../../train/private/hooks.rkt" hook-indexed? hook-named)
         "../../train.rkt")

(provide apply-keywords
         batch-indexed?
         collecting-finalized
         collecting-hparams
         collecting-logger
         collecting-metrics
         collecting-steps
         interval-message
         recorder
         Recorded
         scenario
         scenario-events
         scenario-exn
         toy-loader)

(struct collecting-logger ([metrics #:auto] [hparams #:auto] [finalized #:auto])
  #:auto-value '()
  #:mutable
  #:methods gen:training-logger
  [(define (logger-log-hyperparams! l params)
     (set-collecting-logger-hparams! l (cons params (collecting-logger-hparams l))))
   (define (logger-log-metrics! l metrics step)
     (set-collecting-logger-metrics! l (cons (cons step metrics) (collecting-logger-metrics l))))
   (define (logger-finalize! l status)
     (set-collecting-logger-finalized! l (cons status (collecting-logger-finalized l))))])

(define (collecting-metrics l) (map cdr (collecting-steps l)))
(define (collecting-steps l) (reverse (collecting-logger-metrics l)))
(define (collecting-hparams l) (reverse (collecting-logger-hparams l)))
(define (collecting-finalized l) (reverse (collecting-logger-finalized l)))

(define (apply-keywords proc keyword-values . positional)
  (match (sort keyword-values keyword<? #:key car)
    [(list (cons kws vals) ...) (keyword-apply proc kws vals positional)]))

(define interval-message
  (string-append "an integer #:val-check-interval exceeds the training batches"
                 " #:limit-train-batches selects"))

(struct scenario (skip-epoch fail-at stop-at break-twice? step-scheduler?
                  event-box [exn #:mutable])
  #:name scenario-struct
  #:constructor-name scenario-state)

(define (scenario #:skip-epoch [skip-epoch #f] #:fail-at [fail-at #f]
                  #:stop-at [stop-at #f] #:break-twice? [break-twice? #f]
                  #:step-scheduler? [step-scheduler? #f])
  (scenario-state skip-epoch fail-at stop-at break-twice? step-scheduler?
                  (box '()) #f))

(define (scenario-events sc) (reverse (unbox (scenario-event-box sc))))

(define (toy-loader n)
  (dataloader (tensor-dataset (zeros n 2) (ones n 1)) #:batch-size 2))

(define (batch-indexed? name) (hook-indexed? (hook-named name)))

(define (record! sc source name r)
  (define lr
    (match (run-optimizers r)
      ['() #f]
      [(cons opt _) (learning-rate opt)]))
  (define events (scenario-event-box sc))
  (set-box! events
            (cons (list source name (run-epoch r) (run-global-step r)
                        (and (batch-indexed? name) (run-batch-index r)) lr)
                  (unbox events))))

(define (at? place r)
  (equal? place (list (run-epoch r) (run-batch-index r))))

(define (raise-planned sc)
  (cond
    [(scenario-break-twice? sc) (break-thread (current-thread))]
    [else (error 'training-step "planned failure")]))

(begin-for-syntax
  (define (special source name)
    (case name
      [(on-before-batch-transfer on-after-batch-transfer)
       #'((apply values batch))]
      [(on-train-batch-start)
       (if (eq? source 'trainer)
           #'((when (at? (scenario-skip-epoch sc) run) (skip-epoch! run)))
           #'())]
      [(on-exception)
       #'((set-scenario-exn! sc exn)
          (when (scenario-break-twice? sc) (break-thread (current-thread))))]
      [(state-dict) #'((hasheq))]
      [else #'()]))

  (define (hook-definitions source side?)
    (for/list ([h (in-list definable-hooks)] #:when (side? h))
      (define name (hook-name h))
      (define header
        (if (hook-batch? h)
            #`(#,name #,@(hook-formals h) . batch)
            #`(#,name #,@(hook-formals h))))
      #`(define-hook #,header
          (record! sc '#,source '#,name run)
          #,@(special source name)))))

(define-syntax-parse-rule (define-recorded-trainer name:id)
  #:with (hook ...) (hook-definitions 'trainer trainer-hook?)
  (define-trainer name (net sc)
    (define-init (s) (set! net (Linear 2 1)) (set! sc s))
    (define-callbacks (record! sc 'trainer 'configure-callbacks run) '())
    (define-optimizers
      (record! sc 'trainer 'configure-optimizers run)
      (define opt (sgd (parameters net) #:lr 0.1))
      (define schedule (step-lr opt #:step-size 1 #:gamma 0.5))
      (values opt
              (if (scenario-step-scheduler? sc)
                  (scheduler-config schedule #:interval 'step)
                  schedule)))
    (define-training-step (x y)
      (record! sc 'trainer 'training-step run)
      (when (at? (scenario-fail-at sc) run) (raise-planned sc))
      (when (at? (scenario-stop-at sc) run) (request-stop! run))
      (mse-loss (net x) y))
    (define-validation-step (x y)
      (record! sc 'trainer 'validation-step run)
      (mse-loss (net x) y))
    (define-test-step (x y)
      (record! sc 'trainer 'test-step run)
      (mse-loss (net x) y))
    (define-predict-step (x y)
      (record! sc 'trainer 'predict-step run)
      (net x))
    hook ...))

(define-syntax-parse-rule (define-recorder name:id)
  #:with (hook ...) (hook-definitions 'callback callback-hook?)
  (define-callback name (sc)
    (define-state-key "recorder")
    hook ...))

(define-recorded-trainer Recorded)
(define-recorder recorder)
