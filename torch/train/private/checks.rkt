#lang racket/base

(require (only-in racket/contract/base
                  ->i any any/c flat-named-contract flat-rec-contract list/c listof
                  or/c)
         (only-in racket/list append*)
         (only-in racket/match match)
         (only-in "../../nn/optim.rkt" optimizer?)
         (only-in "../../nn/scheduler.rkt" scheduler-optimizer-of scheduler?)
         (only-in "../../private/contract.rkt" define/contract-out)
         (only-in "../logging.rkt" loss/c)
         (only-in "../run.rkt" state-key/c)
         (only-in "types.rkt"
                  callback? make-scheduler-config scheduler-config-scheduler
                  scheduler-config?))

(provide readable-arguments)

(define (optimizer-only? v)
  (and (optimizer? v) (not (scheduler? v))))

(define (scheduler-of v)
  (cond
    [(scheduler-config? v) (scheduler-config-scheduler v)]
    [(scheduler? v) v]
    [else #f]))

(define ((optimizer-values? automatic?) vs)
  (define optimizers (filter optimizer-only? vs))
  (define schedulers (filter values (map scheduler-of vs)))
  (and (andmap (lambda (v) (or (optimizer? v) (scheduler-config? v))) vs)
       (if automatic? (= 1 (length optimizers)) (pair? optimizers))
       (for/and ([s (in-list schedulers)])
         (and (memq (scheduler-optimizer-of s) optimizers) #t))))

(define (optimizer-values/c automatic?)
  (flat-named-contract
   (if automatic?
       '(one-optimizer-and-its-schedulers/c)
       '(optimizers-and-their-schedulers/c))
   (optimizer-values? automatic?)))

(define (as-config v)
  (if (scheduler-config? v) v (make-scheduler-config v 'epoch 1)))

(define/contract-out (configured-optimizers automatic? vs)
  (->i ([automatic? boolean?] [vs (automatic?) (optimizer-values/c automatic?)])
       any)
  (values (filter optimizer-only? vs)
          (map as-config (filter scheduler-of vs))))

(define step-result/c
  (flat-named-contract
   'training-step-result/c
   (or/c (list/c #f) (list/c loss/c) (list/c loss/c any/c))))

(define/contract-out (training-step-result vs)
  (->i ([vs step-result/c]) any)
  (match vs
    [(list loss) (values loss #f)]
    [(list loss extra) (values loss extra)]))

(define callbacks/c
  (flat-rec-contract callbacks/c (or/c callback? (listof callbacks/c))))

(define (flatten-callbacks v)
  (if (callback? v) (list v) (append* (map flatten-callbacks v))))

(define/contract-out (configured-callbacks vs)
  (->i ([vs (listof callbacks/c)]) [result any/c])
  (flatten-callbacks vs))

(define (readable? v)
  (or (real? v) (string? v) (symbol? v) (boolean? v)
      (and (list? v) (andmap readable? v))))

(define (readable-arguments names vals)
  (for/hasheq ([name (in-list names)] [v (in-list vals)] #:when (readable? v))
    (values name v)))

(define/contract-out (checked-state-key key)
  (->i ([key state-key/c]) [result any/c])
  key)

