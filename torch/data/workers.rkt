#lang racket/base

(require (only-in ffi/unsafe register-finalizer)
         (only-in racket/contract/base -> any/c)
         (only-in racket/match match match-define)
         (only-in "../foreign.rkt"
                  current-generator device device-type make-generator tensor?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "dataset.rkt" collate/c dataset-batch dataset? indices/c))

;; Workers hold the queue side only: their mailbox, the results channel and
;; the stop semaphore. The pool, its reorder buffer and the traversal that
;; holds it stay unreachable from them, so the finalizer can stop them.
(struct worker-pool (threads results stop buffer))

(define/contract-out (start-workers ds collate n base-seed) ;; noqa
  (-> dataset? collate/c exact-positive-integer? exact-nonnegative-integer?
      worker-pool?)
  (error 'start-workers "unimplemented"))

(define/contract-out (hand-off! pool r indices) ;; noqa
  (-> worker-pool? exact-nonnegative-integer? indices/c void?)
  (error 'hand-off! "unimplemented"))

;; ok? #t: the payload is the batch's values as a list; #f: the raised value
(define/contract-out (await-batch pool r) ;; noqa
  (-> worker-pool? exact-nonnegative-integer? (values boolean? any/c))
  (error 'await-batch "unimplemented"))

(define/contract-out (stop-workers! pool) (-> worker-pool? void?) ;; noqa
  (error 'stop-workers! "unimplemented"))
