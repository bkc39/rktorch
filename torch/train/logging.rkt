#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i and/c any/c contract-out listof or/c property/c
                  unsupplied-arg?)
         (only-in racket/generic define-generics)
         (only-in "../foreign.rkt" dtype numel tensor?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "private/hooks.rkt"
                  log-rule-default-on-epoch? log-rule-default-on-step?
                  log-rule-on-epoch log-rule-on-step)
         (only-in "private/types.rkt" define-named-contract not-implemented)
         (only-in "run.rkt" logging-run/c))

(provide gen:logger
         logger-finalize!
         logger-log-hyperparams!
         logger-log-metrics!
         (contract-out [logger? (-> any/c boolean?)]))

(define-generics logger
  (logger-log-hyperparams! logger params)
  (logger-log-metrics! logger metrics step)
  (logger-finalize! logger status))

(define-named-contract reduce/c (or/c 'mean 'sum 'max 'min))

(define-named-contract scalar-tensor/c
  (and/c tensor? (property/c numel 1)))

(define-named-contract metric-value/c (or/c real? scalar-tensor/c))

(define-named-contract loss/c
  (and/c scalar-tensor/c
         (property/c dtype (or/c 'float32 'float64 'float16 'bfloat16))))

(define-named-contract logger-spec/c (or/c #f logger? (listof logger?)))

(module+ private
  (provide host-reads))

(define (host-reads) (not-implemented 'host-reads 'T1))

(define (hook-log-rule r) (not-implemented 'run-log! 'T1))

(define/contract-out (run-log! r key value
                               #:progress? [progress? #f]
                               #:on-step? [on-step? (log-rule-default-on-step?
                                                     (hook-log-rule r))]
                               #:on-epoch? [on-epoch? (log-rule-default-on-epoch?
                                                       (hook-log-rule r))]
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
       (or (unsupplied-arg? on-step?) (memq on-step? (log-rule-on-step (hook-log-rule r))))
       #:pre/name (r on-epoch?) "this hook logs only per epoch"
       (or (unsupplied-arg? on-epoch?) (memq on-epoch? (log-rule-on-epoch (hook-log-rule r))))
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
