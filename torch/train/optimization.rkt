#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i any/c contract-out flat-named-contract hash/c)
         (only-in "../nn/optim.rkt" optimizer?)
         (only-in "../nn/scheduler.rkt" scheduler?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "logging.rkt" loss/c)
         (only-in "private/checks.rkt" checked-hparam)
         (only-in "private/state.rkt" run-optimizers)
         (only-in "private/types.rkt"
                  make-scheduler-config not-implemented scheduler-config-frequency
                  scheduler-config-interval scheduler-config-scheduler scheduler-config?
                  trainer-hook trainer-hparam-names trainer?)
         (only-in "run.rkt" interval/c manual-run/c))

(provide (contract-out
          [scheduler-config? (-> any/c boolean?)]
          [scheduler-config-scheduler (-> scheduler-config? scheduler?)]
          [scheduler-config-interval (-> scheduler-config? interval/c)]
          [scheduler-config-frequency (-> scheduler-config? exact-positive-integer?)]))

(define (hparam-name/c tr)
  (flat-named-contract
   'hparam-name/c
   (lambda (name) (and (memq name (trainer-hparam-names tr)) #t))))

(define/contract-out (scheduler-config s #:interval [interval 'epoch]
                                       #:frequency [frequency 1])
  (->* (scheduler?) (#:interval interval/c #:frequency exact-positive-integer?)
       scheduler-config?)
  (make-scheduler-config s interval frequency))

(define (configured? r opt)
  (and (memq opt (run-optimizers r)) #t))

(define (hparam-table tr)
  ((trainer-hook tr 'hyperparameters) tr))

(define/contract-out (hparam tr name)
  (->i ([tr trainer?] [name (tr) (hparam-name/c tr)]) [_ real?])
  (checked-hparam 'hparam name (hash-ref (hparam-table tr) name)))

(define/contract-out (set-hparam! tr name v)
  (->i ([tr trainer?] [name (tr) (hparam-name/c tr)] [v real?]) [_ void?])
  ((trainer-hook tr 'set-hyperparameter!) tr name v)
  (void))

(define/contract-out (trainer-hparams tr)
  (-> trainer? (hash/c symbol? real? #:immutable #t))
  (define all (hparam-table tr))
  (for/hasheq ([name (in-list (trainer-hparam-names tr))])
    (values name (checked-hparam 'trainer-hparams name (hash-ref all name)))))

(define/contract-out (manual-backward! r loss)
  (-> manual-run/c loss/c void?)
  (not-implemented 'manual-backward! 'T2))

(define/contract-out (manual-step! r opt)
  (->i ([r manual-run/c] [opt optimizer?])
       #:pre/name (r opt) "opt is not one of define-optimizers' optimizers"
       (configured? r opt)
       [_ void?])
  (not-implemented 'manual-step! 'T2))

(define/contract-out (manual-zero-grads! r opt)
  (->i ([r manual-run/c] [opt optimizer?])
       #:pre/name (r opt) "opt is not one of define-optimizers' optimizers"
       (configured? r opt)
       [_ void?])
  (not-implemented 'manual-zero-grads! 'T2))
