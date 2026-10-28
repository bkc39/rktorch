#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/syntax format-id)
                     (only-in syntax/parse/pre id)
                     (only-in "private/hooks.rkt"
                              callback-hook? definable-hooks hook-batch?
                              hook-formals hook-name))
         (only-in racket/contract/base ->* or/c procedure-arity-includes/c)
         (only-in racket/match match)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "private/types.rkt" callback-info callback? prop:callback)
         (only-in "run.rkt" state-key/c))

(begin-for-syntax
  (define callback-hooks (filter callback-hook? definable-hooks))

  (define (hook-keyword h) (string->keyword (symbol->string (hook-name h))))

  (define (hook-procedure/c h)
    (if (hook-batch? h)
        #'procedure?
        #`(procedure-arity-includes/c #,(add1 (length (hook-formals h)))))))

(struct lambda-hooks (procs key)
  #:property prop:callback
  (callback-info 'plain
                 'lambda-callback
                 (lambda (cb name)
                   (match (hash-ref (lambda-hooks-procs cb) name #f)
                     [#f #f]
                     [p (lambda (_cb r . args) (apply p r args))]))
                 (lambda (cb) (lambda-hooks-key cb))))

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
