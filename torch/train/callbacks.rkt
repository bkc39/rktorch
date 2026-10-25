#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/syntax format-id)
                     ;; whole-module on purpose: the template needs ~@
                     syntax/parse/pre
                     (only-in "private/hooks.rkt"
                              callback-hook? definable-hooks hook-batch?
                              hook-formals hook-name))
         (only-in racket/contract/base
                  -> ->* any/c contract-out or/c procedure-arity-includes/c)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "private/types.rkt" callback? not-implemented trainer?)
         (only-in "run.rkt" state-key/c))

(provide (contract-out [callback? (-> any/c boolean?)]
                       [trainer? (-> any/c boolean?)]))

(begin-for-syntax
  (define (hook-procedure/c h)
    (if (hook-batch? h)
        #'procedure?
        #`(procedure-arity-includes/c #,(add1 (length (hook-formals h)))))))

(define-syntax (define-lambda-callback stx)
  (syntax-parse stx
    [(_ name:id)
     #:do [(define hs (filter callback-hook? definable-hooks))]
     #:with (kw ...)
     (for/list ([h (in-list hs)])
       (string->keyword (symbol->string (hook-name h))))
     #:with (proc ...)
     (for/list ([h (in-list hs)]) (format-id #'name "~a" (hook-name h)))
     #:with (proc/c ...) (map hook-procedure/c hs)
     #'(define/contract-out (name #:state-key [state-key "lambda-callback"]
                                  (~@ kw [proc #f]) ...)
         (->* () (#:state-key state-key/c (~@ kw (or/c #f proc/c)) ...)
              callback?)
         (not-implemented 'name 'T1))]))

(define-lambda-callback lambda-callback)
