#lang racket/base

(require (only-in racket/contract/base
                  -> ->* any/c hash/c or/c unconstrained-domain->)
         (only-in "../foreign.rkt" device?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "../private/not-implemented.rkt" not-implemented)
         (only-in "private/types.rkt" define-named-contract trainer?)
         (only-in "private/value.rkt" checkpoint-value?)
         (only-in "run.rkt" run?))

(define-named-contract checkpoint-value/c checkpoint-value?)

(define/contract-out (save-checkpoint! r path #:weights-only? [weights-only? #f]) ;; noqa
  (->* (run? path-string?) (#:weights-only? boolean?) void?)
  (not-implemented 'save-checkpoint! 'T3))

(define/contract-out (read-checkpoint path) ;; noqa
  (-> path-string? (hash/c string? checkpoint-value/c #:immutable #t))
  (not-implemented 'read-checkpoint 'T3))

(define/contract-out (load-from-checkpoint make-trainer path ;; noqa
                                           #:device [device #f]
                                           #:strict? [strict? #t]
                                           #:init-args [init-args (hasheq)])
  (->* ((unconstrained-domain-> trainer?) path-string?)
       (#:device (or/c #f device?)
        #:strict? boolean?
        #:init-args (hash/c symbol? any/c #:immutable #t))
       trainer?)
  (not-implemented 'load-from-checkpoint 'T3))
