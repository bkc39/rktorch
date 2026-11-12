#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/syntax format-id)
                     (only-in syntax/parse/pre id)
                     (only-in "private/hooks.rkt"
                              callback-hook? definable-hooks hook-batch?
                              hook-formals hook-name))
         (only-in racket/contract/base
                  -> ->* and/c contract? flat-named-contract hash/c
                  not/c or/c procedure-arity-includes/c)
         (only-in racket/match match)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "private/state.rkt" run-trainer)
         (only-in "private/types.rkt"
                  callback-property callback? prop:callback trainer-automatic?)
         (only-in "run.rkt" run-accumulate-grad-batches run-epoch state-key/c)
         (only-in (submod "run.rkt" private) set-run-accumulate-grad-batches!))

(begin-for-syntax
  (define callback-hooks (filter callback-hook? definable-hooks))

  (define (hook-keyword h) (string->keyword (symbol->string (hook-name h))))

  (define (hook-procedure/c h)
    (if (hook-batch? h)
        #'procedure?
        #`(procedure-arity-includes/c #,(add1 (length (hook-formals h)))))))

(define (lambda-hook cb name)
  (match (hash-ref (lambda-hooks-procs cb) name #f)
    [#f #f]
    [p (lambda (_cb r . args) (apply p r args))]))

(struct lambda-hooks (procs key)
  #:property prop:callback
  (callback-property 'plain 'lambda-callback
                     #:hook-ref lambda-hook
                     #:state-key (lambda (cb) (lambda-hooks-key cb))))

(define (hooks-table pairs)
  (for/hasheq ([pair (in-list pairs)] #:when (cdr pair))
    (values (car pair) (cdr pair))))

(define-syntax-parse-rule (define-lambda-callback name:id)
  #:with (kw ...) (map hook-keyword callback-hooks)
  #:with (proc ...) (map (lambda (h) (format-id #'name "~a" (hook-name h)))
                         callback-hooks)
  #:with (proc/c ...) (map hook-procedure/c callback-hooks)
  (define/contract-out (name #:state-key [state-key "lambda-callback"]
                             (~@ kw [proc #f]) ...)
    (->* () (#:state-key state-key/c (~@ kw (or/c #f proc/c)) ...) callback?)
    (lambda-hooks (hooks-table (list (cons 'proc proc) ...)) state-key)))

(define-lambda-callback lambda-callback)

(define/contract-out accumulation-scheduling/c
  contract?
  (flat-named-contract 'accumulation-scheduling/c
                       (and/c (hash/c exact-nonnegative-integer? exact-positive-integer?
                                      #:immutable #t)
                              (not/c hash-empty?))))

(define (factor-at scheduling epoch)
  (match (filter (lambda (e) (<= e epoch)) (hash-keys scheduling))
    ['() 1]
    [started (hash-ref scheduling (apply max started))]))

(define (check-schedulable! r)
  (cond
    [(not (trainer-automatic? (run-trainer r)))
     (raise-arguments-error
      'gradient-accumulation-scheduler
      (string-append "automatic gradient accumulation and the gradient-accumulation-scheduler"
                     " are not supported for manual optimization; remove the callback or"
                     " switch to automatic optimization"))]
    [(not (= 1 (run-accumulate-grad-batches r)))
     (raise-arguments-error
      'gradient-accumulation-scheduler
      (string-append "you have set #:accumulate-grad-batches and are using the"
                     " gradient-accumulation-scheduler callback; either remove"
                     " #:accumulate-grad-batches from fit or remove the callback")
      "#:accumulate-grad-batches" (run-accumulate-grad-batches r))]
    [else (void)]))

(define (accumulation-hook cb name)
  (match name
    ['on-train-start (lambda (_cb r) (check-schedulable! r))]
    ['on-train-epoch-start
     (lambda (cb r)
       (set-run-accumulate-grad-batches!
        r (factor-at (accumulation-scheduler-scheduling cb) (run-epoch r))))]
    [_ #f]))

(struct accumulation-scheduler (scheduling)
  #:property prop:callback
  (callback-property 'plain 'gradient-accumulation-scheduler
                     #:hook-ref accumulation-hook
                     #:state-key (lambda (_cb) "gradient-accumulation-scheduler")))

(define/contract-out (gradient-accumulation-scheduler scheduling)
  (-> accumulation-scheduling/c callback?)
  (accumulation-scheduler scheduling))
