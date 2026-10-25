#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/contract/base
                  -> ->* ->i and/c any/c contract? contract-out
                  flat-named-contract listof or/c unsupplied-arg?)
         (only-in racket/generic define-generics)
         (only-in racket/stxparam define-syntax-parameter)
         (only-in "../foreign.rkt" dtype numel tensor?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "private/types.rkt" not-implemented)
         (only-in "run.rkt" logging-run/c))

(provide gen:logger
         log!
         logger-finalize!
         logger-log-hyperparams!
         logger-log-metrics!
         run
         (contract-out [logger? (-> any/c boolean?)]))

(define-syntax-parameter log!
  (lambda (stx)
    (raise-syntax-error #f "only allowed inside a trainer or callback hook body"
                        stx)))

(define-syntax-parameter run
  (lambda (stx)
    (raise-syntax-error #f "only allowed inside a trainer or callback hook body"
                        stx)))

(define-generics logger
  (logger-log-hyperparams! logger params)
  (logger-log-metrics! logger metrics step)
  (logger-finalize! logger status))

(define/contract-out reduce/c contract?
  (flat-named-contract 'reduce/c (or/c 'mean 'sum 'max 'min)))

(define/contract-out scalar-tensor/c contract?
  (flat-named-contract 'scalar-tensor/c
                       (and/c tensor? (lambda (t) (= 1 (numel t))))))

(define/contract-out metric-value/c contract?
  (flat-named-contract 'metric-value/c (or/c real? scalar-tensor/c)))

(define/contract-out loss/c contract?
  (flat-named-contract
   'loss/c
   (and/c scalar-tensor/c
          (lambda (t) (and (memq (dtype t) '(float32 float64 float16 bfloat16))
                           #t)))))

(define/contract-out logger-spec/c contract?
  (flat-named-contract 'logger-spec/c (or/c #f logger? (listof logger?))))

(module+ private
  (provide host-reads))

(define (host-reads) (not-implemented 'host-reads 'T1))

(define (hook-default-on-step r) (not-implemented 'run-log! 'T1))
(define (hook-default-on-epoch r) (not-implemented 'run-log! 'T1))
(define (hook-logs-per? r interval) (not-implemented 'run-log! 'T1))

(define (allowed? r flag interval)
  (or (unsupplied-arg? flag) (not flag) (hook-logs-per? r interval)))

(define/contract-out (run-log! r key value
                               #:progress? [progress? #f]
                               #:on-step? [on-step? (hook-default-on-step r)]
                               #:on-epoch? [on-epoch? (hook-default-on-epoch r)]
                               #:reduce [reduce 'mean]
                               #:batch-size [batch-size #f]
                               #:logger? [logger? #t])
  (->i ([r logging-run/c] [key symbol?] [value metric-value/c])
       (#:progress? [progress? boolean?]
        #:on-step? [on-step? boolean?]
        #:on-epoch? [on-epoch? boolean?]
        #:reduce [reduce reduce/c]
        #:batch-size [batch-size (or/c #f exact-positive-integer?)]
        #:logger? [logger? boolean?])
       #:pre/name (r on-step?) "this hook logs only per epoch"
       (allowed? r on-step? 'step)
       #:pre/name (r on-epoch?) "this hook logs only per step"
       (allowed? r on-epoch? 'epoch)
       [result void?])
  (not-implemented 'run-log! 'T1))

(define/contract-out (stdout-logger #:out [out (current-output-port)])
  (->* () (#:out output-port?) logger?)
  (not-implemented 'stdout-logger 'T1))

(define/contract-out (jsonl-logger dir #:exists [exists 'error])
  (->* (path-string?) (#:exists (or/c 'error 'append 'replace)) logger?)
  (not-implemented 'jsonl-logger 'T1))

(define/contract-out (csv-logger dir #:exists [exists 'error])
  (->* (path-string?) (#:exists (or/c 'error 'append 'replace)) logger?)
  (not-implemented 'csv-logger 'T1))
