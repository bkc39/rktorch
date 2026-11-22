#lang racket/base

(require (only-in racket/contract/base
                  -> ->* any/c hash/c or/c unconstrained-domain->)
         (only-in "../foreign.rkt" device?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "private/types.rkt" define-named-contract trainer?)
         (only-in "private/value.rkt" checkpoint-value?)
         (only-in "run.rkt" run?))

(define-named-contract checkpoint-value/c checkpoint-value?)

(define/contract-out (save-checkpoint! r path #:weights-only? [weights-only? #f]) ;; noqa
  (->* (run? path-string?) (#:weights-only? boolean?) void?)
  (error 'unimplemented))

(define/contract-out (read-checkpoint path) ;; noqa
  (-> path-string? (hash/c string? checkpoint-value/c #:immutable #t))
  (error 'unimplemented))

(define/contract-out (load-from-checkpoint make-trainer path ;; noqa
                                           #:device [device #f]
                                           #:strict? [strict? #t]
                                           #:init-args [overrides (hasheq)])
  (->* ((unconstrained-domain-> trainer?) path-string?)
       (#:device (or/c #f device?)
        #:strict? boolean?
        #:init-args (hash/c symbol? any/c #:immutable #t))
       trainer?)
  (error 'unimplemented))
