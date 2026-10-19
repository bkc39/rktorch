#lang racket/base

(require (only-in racket/contract/base -> any/c not/c)
         (only-in "../private/contract.rkt" define/contract-out))

;; A table and a counting semaphore, so a worker's put never waits for the main thread.
(struct results-queue (table ready) ;; noqa
  #:constructor-name make-results-queue
  #:omit-define-syntaxes)

(provide results-queue?)

(define/contract-out (results-queue) (-> results-queue?) ;; noqa
  (make-results-queue (make-hasheqv) (make-semaphore 0)))

(define/contract-out (results-queue-put! q r v) ;; noqa
  (-> results-queue? exact-nonnegative-integer? (not/c not) void?)
  (hash-set! (results-queue-table q) r v)
  (semaphore-post (results-queue-ready q)))

(define/contract-out (results-queue-take! q r abandon breakable?) ;; noqa
  (-> results-queue? exact-nonnegative-integer? evt? boolean? any/c)
  (define table (results-queue-table q))
  (define ready (wrap-evt (results-queue-ready q) (lambda (_) #f)))
  (define abandoned (wrap-evt abandon (lambda _results #t)))
  (define wait (if breakable? sync/enable-break sync))
  (let loop ([abandoned? #f])
    (define v (hash-ref table r #f))
    (cond
      [v
       (hash-remove! table r)
       v]
      [abandoned? #f]
      [else (loop (wait ready abandoned))])))
