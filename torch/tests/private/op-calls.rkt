#lang racket/base

(require (only-in "../../foreign/raw/device-queries.rkt" allocator-reading)
         (only-in "../../foreign/raw/fault.rkt" with-crossing-hook)
         (only-in "../../foreign/raw/pressure-settings.rkt"
                  native-collect-at-troughs)
         (only-in "race-harness.rkt" with-race-hook))

(provide op-calls-of)

(define ((uncounted reading) dev)
  (with-crossing-hook #f
    (reading dev)))

(define (op-calls-of thunk)
  (define here (current-thread))
  (define crossings 0)
  (define frees 0)
  (define tensors 0)
  (parameterize ([native-collect-at-troughs #f]
                 [allocator-reading (uncounted (allocator-reading))])
    (with-crossing-hook (lambda ()
                          (when (eq? (current-thread) here)
                            (set! crossings (add1 crossings))))
      (with-race-hook (lambda (name _subject)
                        (case (and (eq? (current-thread) here) name)
                          [(ledger-entry-added) (set! tensors (add1 tensors))]
                          [(free-released finalizer-releasing)
                           (set! frees (add1 frees))]
                          [else (void)]))
        (thunk))))
  (- crossings frees (* 2 tensors)))
