#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i any/c contract-out flat-named-contract hash/c)
         (only-in "../nn/optim.rkt" optimizer?)
         (only-in "../nn/scheduler.rkt" scheduler?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "logging.rkt" loss/c)
         (only-in "private/types.rkt"
                  not-implemented scheduler-config? trainer-hparam-names
                  trainer?)
         (only-in "run.rkt" interval/c manual-run/c))

(provide (contract-out [scheduler-config? (-> any/c boolean?)]))

(define (hparam-name/c tr)
  (flat-named-contract
   'hparam-name/c
   (lambda (name) (and (memq name (trainer-hparam-names tr)) #t))))

(define/contract-out (scheduler-config s #:interval [interval 'epoch]
                                       #:frequency [frequency 1])
  (->* (scheduler?) (#:interval interval/c #:frequency exact-positive-integer?)
       scheduler-config?)
  (not-implemented 'scheduler-config 'T2))

(define/contract-out (hparam tr name)
  (->i ([tr trainer?] [name (tr) (hparam-name/c tr)]) [_ real?])
  (not-implemented 'hparam 'T2))

(define/contract-out (set-hparam! tr name v)
  (->i ([tr trainer?] [name (tr) (hparam-name/c tr)] [v real?]) [_ void?])
  (not-implemented 'set-hparam! 'T2))

(define/contract-out (trainer-hparams tr)
  (-> trainer? (hash/c symbol? real? #:immutable #t))
  (not-implemented 'trainer-hparams 'T2))

(define/contract-out (manual-backward! r loss)
  (-> manual-run/c loss/c void?)
  (not-implemented 'manual-backward! 'T2))

(define/contract-out (manual-step! r opt)
  (-> manual-run/c optimizer? void?)
  (not-implemented 'manual-step! 'T2))

(define/contract-out (manual-zero-grads! r opt)
  (-> manual-run/c optimizer? void?)
  (not-implemented 'manual-zero-grads! 'T2))
