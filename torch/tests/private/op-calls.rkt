#lang racket/base

(require (only-in "../../foreign/raw/device-queries.rkt" allocator-reading)
         (only-in "../../foreign/raw/fault.rkt" with-crossing-hook)
         (only-in "../../foreign/raw/pressure-settings.rkt"
                  native-collect-at-troughs)
         (only-in "race-harness.rkt" with-race-hook))

(provide crossings-of
         faulted-at?
         op-calls-of)

(define ((uncounted reading) dev)
  (with-crossing-hook #f
    (reading dev)))

(define (call-with-own-crossings on-crossing on-race thunk)
  (define here (current-thread))
  (define reaping? #f)
  (with-crossing-hook (lambda ()
                        (cond
                          [(not (eq? (current-thread) here)) (void)]
                          [reaping? (set! reaping? #f)]
                          [else (on-crossing)]))
    (with-race-hook (lambda (name subject)
                      (cond
                        [(not (eq? (current-thread) here)) (void)]
                        [(memq name '(finalizer-releasing generator-releasing))
                         (set! reaping? #t)]
                        [else (on-race name subject)]))
      (thunk))))

(define (batch-crossings held)
  (if (null? held) 0 (- 1 (length held))))

(define (tally thunk)
  (define crossings 0)
  (define accounting 0)
  (define (count! n) (set! accounting (+ accounting n)))
  (parameterize ([native-collect-at-troughs #f]
                 [allocator-reading (uncounted (allocator-reading))])
    (call-with-own-crossings
     (lambda () (set! crossings (add1 crossings)))
     (lambda (name subject)
       (case name
         [(ledger-entry-added) (count! 2)]
         [(ledger-entry-added/batched) (count! -2)]
         [(free-released) (count! 1)]
         [(frees-released/batched) (count! (batch-crossings subject))]
         [else (void)]))
     thunk))
  (values crossings accounting))

(define (crossings-of thunk)
  (define-values (crossings _accounting) (tally thunk))
  crossings)

(define (op-calls-of thunk)
  (define-values (crossings accounting) (tally thunk))
  (- crossings accounting))

(struct injected (k))

(define (faulted-at? k thunk)
  (define n 0)
  (with-handlers ([injected? (lambda (_) #t)])
    (call-with-own-crossings (lambda ()
                               (set! n (add1 n))
                               (when (= n k)
                                 (raise (injected k))))
                             void
                             thunk)
    #f))
