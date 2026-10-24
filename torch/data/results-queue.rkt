#lang racket/base

(require (only-in racket/contract/base -> and/c any/c not/c)
         (only-in racket/function const)
         (only-in "../private/contract.rkt" define/contract-out))

;; A worker's put never waits for the main thread. Closing detaches the
;; table, so a put that races the close lands in a table nothing else reaches.
(struct results-queue ([table #:mutable] ready) ;; noqa
  #:constructor-name make-results-queue
  #:omit-define-syntaxes)

(provide batch-index/c
         results-queue?)

(define batch-index/c exact-nonnegative-integer?)

(define (open? q)
  (and (results-queue-table q) #t))

(define/contract-out (results-queue) (-> results-queue?) ;; noqa
  (make-results-queue (make-hasheqv) (make-semaphore 0)))

(define/contract-out (results-queue-put! q index result) ;; noqa
  (-> results-queue? batch-index/c (not/c not) void?)
  (define table (results-queue-table q))
  (when table
    (hash-set! table index result))
  (semaphore-post (results-queue-ready q)))

(define/contract-out (results-queue-take! q index ;; noqa
                                          #:abandon abandon
                                          #:breakable? breakable?)
  (-> (and/c results-queue? open?) batch-index/c #:abandon evt? #:breakable? boolean?
      any/c)
  (define table (results-queue-table q))
  (define ready (wrap-evt (results-queue-ready q) (const #f)))
  (define abandoned (wrap-evt abandon (const #t)))
  (define wait (if breakable? sync/enable-break sync))
  (let loop ([abandoned? #f])
    (define result (hash-ref table index #f))
    (cond
      [result
       (hash-remove! table index)
       result]
      [abandoned? #f]
      [else (loop (wait ready abandoned))])))

(define/contract-out (results-queue-close! q) (-> results-queue? void?) ;; noqa
  (define table (results-queue-table q))
  (set-results-queue-table! q #f)
  (when table
    (hash-clear! table)))
