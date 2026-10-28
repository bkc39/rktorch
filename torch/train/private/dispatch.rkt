#lang racket/base

(require (for-syntax racket/base (only-in syntax/parse/pre ...+ expr id))
         (only-in racket/list check-duplicates filter-not partition)
         (only-in racket/match match match*)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../../foreign.rkt" cpu-device device shape tensor? to)
         (only-in "../../nn/layer.rkt" buffers parameters)
         (only-in "../logging.rkt" logger-finalize! logger-log-metrics!)
         (only-in "book.rkt" read-epoch! read-steps!)
         (only-in "hooks.rkt" hook-named hook-order hook-sides)
         (only-in "state.rkt"
                  run-book run-callbacks run-global-step run-hook run-loggers
                  run-set-up? run-torn-down? run-trainer set-run-active?!
                  set-run-batch-size! set-run-callbacks! set-run-hook!
                  set-run-set-up?! set-run-torn-down?!)
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
         with-failure-protocol)

(define-logger rktorch-train)

(define (in-hook r name thunk)
  (define outer (run-hook r))
  (set-run-hook! r name)
  (begin0 (thunk) (set-run-hook! r outer)))

(define (call-trainer r name . args)
  (define tr (run-trainer r))
  (define proc (trainer-hook tr name))
  (and proc (in-hook r name (lambda () (apply proc tr r args)))))

(define (call-trainer/values r name args)
  (define tr (run-trainer r))
  (define proc (trainer-hook tr name))
  (and proc
       (in-hook r name (lambda () (call-with-values (lambda () (apply proc tr r args)) list)))))

(define (call-callback r cb name args)
  (define proc (callback-hook cb name))
  (and proc (in-hook r name (lambda () (apply proc cb r args)))))

(define (call-callbacks r name cbs . args)
  (for ([cb (in-list cbs)])
    (call-callback r cb name args)))

(define (late? cb) (memq (callback-kind cb) '(monitor checkpoint)))

(define (call-hooks r name #:between [between void] . args)
  (define h (hook-named name))
  (define cbs (if (memq 'callback (hook-sides h)) (run-callbacks r) '()))
  (define (trainer!) (apply call-trainer r name args))
  (match (hook-order h)
    ['by-kind
     (define-values (late early) (partition late? cbs))
     (apply call-callbacks r name early args)
     (trainer!)
     (between)
     (apply call-callbacks r name late args)]
    ['trainer-first
     (trainer!)
     (apply call-callbacks r name cbs args)]
    [_
     (apply call-callbacks r name cbs args)
     (trainer!)]))

(define (moved v dev)
  (cond
    [(tensor? v) (to v dev)]
    [(pair? v) (cons (moved (car v) dev) (moved (cdr v) dev))]
    [else v]))

(define (transfer r dev batch)
  (define before (or (call-trainer/values r 'on-before-batch-transfer batch) batch))
  (define arrived (moved before dev))
  (or (call-trainer/values r 'on-after-batch-transfer arrived) arrived))

(define (note-batch-size! r batch)
  (set-run-batch-size!
   r
   (match (findf tensor? batch)
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

(define (publish! r metrics)
  (match metrics
    [(? hash-empty?) (void)]
    [_ (for ([l (in-list (run-loggers r))])
         (logger-log-metrics! l metrics (run-global-step r)))]))

(define (publish-steps! r)
  (publish! r (read-steps! (run-book r))))

(define (publish-epoch! r loop)
  (publish! r (read-epoch! (run-book r) loop)))

(define (finish! r status)
  (for ([l (in-list (run-loggers r))])
    (logger-finalize! l status))
  (set-run-active?! r #f))

(define (logged-attempt what thunk)
  (with-handlers ([exn:fail? (lambda (e)
                               (log-rktorch-train-warning "~a raised: ~a" what (exn-message e)))])
    (thunk)))

(define (recover! r stage e)
  (parameterize-break #f
    (for ([cb (in-list (run-callbacks r))])
      (logged-attempt 'on-exception (lambda () (call-callback r cb 'on-exception (list e)))))
    (for ([l (in-list (run-loggers r))])
      (logged-attempt 'logger-finalize! (lambda () (logger-finalize! l 'failed))))
    (match* ((run-set-up? r) (run-torn-down? r))
      [(#t #f) (logged-attempt 'teardown (lambda () (tear-down! r stage)))]
      [(_ _) (void)])
    (set-run-active?! r #f)
    (with-handlers ([exn:break? void])
      (parameterize-break #t (sleep 0)))))

(define (fault? v) (or (exn:fail? v) (exn:break? v)))

(define (call-with-failure-protocol r stage thunk)
  (with-handlers ([fault? (lambda (e) (recover! r stage e) (raise e))])
    (thunk)))

(define-syntax-parse-rule (with-failure-protocol r:expr stage:expr body:expr ...+)
  (call-with-failure-protocol r stage (lambda () body ...)))
