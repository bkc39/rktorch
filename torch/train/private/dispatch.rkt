#lang racket/base

(require (for-syntax racket/base (only-in syntax/parse/pre ...+ expr id))
         (only-in racket/list check-duplicates filter-not partition)
         (only-in racket/match match)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../../foreign.rkt" cpu-device device shape tensor? to)
         (only-in "../../foreign/raw/fault.rkt" native-fault? note-native-fault!)
         (only-in "../../nn/layer.rkt" buffers layer? parameters)
         (only-in (submod "../../nn/mode.rkt" private) eval-layer-count)
         (only-in "../logging.rkt" logger-finalize! logger-log-metrics!)
         (only-in "book.rkt" read-epoch! read-steps!)
         (only-in "hooks.rkt" hook-named hook-order hook-sides)
         (only-in "state.rkt"
                  run-book run-callbacks run-epoch run-hook run-loggers
                  run-set-up? run-torn-down? run-trainer set-run-batch-size!
                  set-run-callbacks! set-run-finished?! set-run-hook! set-run-set-up?!
                  set-run-torn-down?!)
         (only-in "types.rkt"
                  callback-hook callback-kind callback-state-key callback-type
                  trainer-hook))

(provide call-callbacks
         call-hooks
         call-trainer
         call-trainer/values
         finish!
         install-callbacks!
         note-batch-size!
         publish-epoch!
         publish-steps!
         set-up!
         tear-down!
         trainer-device
         transfer
         warn-eval-layers!
         with-failure-protocol)

(define-logger rktorch-train)

(define (call-in-hook r name thunk)
  (define outer (run-hook r))
  (set-run-hook! r name)
  (begin0 (thunk) (set-run-hook! r outer)))

(define (call-trainer r name . args)
  (define tr (run-trainer r))
  (define proc (trainer-hook tr name))
  (and proc (call-in-hook r name (lambda () (apply proc tr r args)))))

(define (call-trainer/values r name args)
  (define tr (run-trainer r))
  (define proc (trainer-hook tr name))
  (and proc
       (call-in-hook r name (lambda () (call-with-values (lambda () (apply proc tr r args)) list)))))

(define (call-callback r cb name args)
  (define proc (callback-hook cb name))
  (and proc (call-in-hook r name (lambda () (apply proc cb r args)))))

(define (call-callbacks r name cbs . args)
  (for ([cb (in-list cbs)])
    (call-callback r cb name args)))

(define (late? cb) (memq (callback-kind cb) '(monitor checkpoint)))

(define (call-hooks r name . args)
  (define h (hook-named name))
  (define cbs (if (memq 'callback (hook-sides h)) (run-callbacks r) '()))
  (define (trainer!) (apply call-trainer r name args))
  (match (hook-order h)
    ['by-kind
     (define-values (late early) (partition late? cbs))
     (apply call-callbacks r name early args)
     (trainer!)
     (apply call-callbacks r name late args)]
    ['trainer-first
     (trainer!)
     (apply call-callbacks r name cbs args)]
    [_
     (apply call-callbacks r name cbs args)
     (trainer!)]))

(define (moved v dev)
  (match v
    [(? tensor?) (to v dev)]
    [(cons a d) (cons (moved a dev) (moved d dev))]
    [_ v]))

(define (transfer r dev batch)
  (define before (or (call-trainer/values r 'on-before-batch-transfer batch) batch))
  (define arrived (moved before dev))
  (or (call-trainer/values r 'on-after-batch-transfer arrived) arrived))

(define (leading-tensor v)
  (match v
    [(? tensor?) v]
    [(cons a d) (or (leading-tensor a) (leading-tensor d))]
    [_ #f]))

(define (note-batch-size! r batch)
  (set-run-batch-size!
   r
   (match (leading-tensor batch)
     [#f 1]
     [t (match (shape t)
          ['() 1]
          [(cons rows _) rows])])))

(define (trainer-device tr)
  (match (append (parameters tr) (buffers tr))
    ['() (cpu-device)]
    [(cons t _) (device t)]))

(define (stateful-key-clash cbs)
  (check-duplicates (map callback-state-key
                         (filter (lambda (cb) (callback-hook cb 'state-dict)) cbs))))

(define (install-callbacks! r who passed #:fast? fast?)
  (define own (or (call-trainer r 'configure-callbacks) '()))
  (define own-types (map callback-type own))
  (define merged
    (append own (filter-not (lambda (cb) (memq (callback-type cb) own-types)) passed)))
  (define-values (checkpoints others)
    (partition (lambda (cb) (eq? (callback-kind cb) 'checkpoint)) merged))
  (define ordered (filter-not (lambda (cb) (and fast? (late? cb))) (append others checkpoints)))
  (match (stateful-key-clash ordered)
    [#f (set-run-callbacks! r ordered)]
    [key (raise-arguments-error
          who (format "two stateful callbacks share the state key ~s" key))]))

(define (set-up! r stage)
  (set-run-set-up?! r #t)
  (call-hooks r 'setup stage))

(define (tear-down! r stage)
  (set-run-torn-down?! r #t)
  (call-hooks r 'teardown stage))

(define (publish! r metrics step #:epoch? epoch?)
  (define row
    (if (and epoch? (not (hash-has-key? metrics 'epoch)))
        (hash-set metrics 'epoch (run-epoch r))
        metrics))
  (unless (hash-empty? metrics)
    (for ([l (in-list (run-loggers r))])
      (logger-log-metrics! l row step))))

(define (publish-steps! r loop step)
  (publish! r (read-steps! (run-book r) loop) step #:epoch? (eq? loop 'train)))

(define (publish-epoch! r loop step)
  (publish! r (read-epoch! (run-book r) loop) step #:epoch? #t))

(define (finish! r status)
  (set-run-finished?! r #t)
  (for ([l (in-list (run-loggers r))])
    (logger-finalize! l status)))

(define (warn-eval-layers! r)
  (define n (match (run-trainer r) [(? layer? tr) (eval-layer-count tr)] [_ 0]))
  (when (positive? n)
    (define message
      (format "fit: ~a layer(s) in eval mode at the start of training; they stay in eval mode" n))
    (log-rktorch-train-warning message)
    (eprintf "rktorch-train: ~a\n" message)))

(define (logged-attempt what thunk)
  (with-handlers ([exn:fail? (lambda (e)
                               (when (native-fault? e)
                                 (note-native-fault! (format "running ~a during failure recovery" what)))
                               (log-rktorch-train-warning "~a raised: ~a" what (exn-message e)))])
    (thunk)))

(define (recover! r stage e absorb-break?)
  (parameterize-break #f
    (for ([cb (in-list (run-callbacks r))])
      (logged-attempt 'on-exception (lambda () (call-callback r cb 'on-exception (list e)))))
    (for ([l (in-list (run-loggers r))])
      (logged-attempt 'logger-finalize! (lambda () (logger-finalize! l 'failed))))
    (when (and (run-set-up? r) (not (run-torn-down? r)))
      (logged-attempt 'teardown (lambda () (tear-down! r stage))))
    (set-run-finished?! r #t)
    (when absorb-break?
      (with-handlers ([exn:break? void])
        (parameterize-break #t (sleep 0))))))

(define (fault? v) (or (exn:fail? v) (exn:break? v)))

(define (call-with-failure-protocol r stage thunk)
  (define breaks? (break-enabled))
  (with-handlers ([fault? (lambda (e) (recover! r stage e breaks?) (raise e))])
    (thunk)))

(define-syntax-parse-rule (with-failure-protocol r:expr stage:expr body:expr ...+)
  (call-with-failure-protocol r stage (lambda () body ...)))
