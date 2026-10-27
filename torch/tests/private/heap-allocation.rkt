#lang racket/base

(require (only-in "../../foreign.rkt" native-memory-unaccounted native-memory-use))

(provide median-allocation)

;; Every native tensor charges the collector its size as phantom bytes, and
;; (current-memory-use 'cumulative) counts those; the ledger holds the same
;; figure, so its change is subtracted to leave the Racket heap alone.
(define (phantom-bytes)
  (for/sum ([entry (in-list (append (native-memory-use) (native-memory-unaccounted)))])
    (cdr entry)))

(define (allocation thunk)
  (collect-garbage 'minor)
  (define phantom (phantom-bytes))
  (define heap (current-memory-use 'cumulative))
  (call-with-values thunk void)
  (define heap-after (current-memory-use 'cumulative))
  (- heap-after heap (- (phantom-bytes) phantom)))

(define (median-allocation thunk)
  (call-with-values thunk void)
  (define runs
    (sort (for/list ([_ (in-range 15)])
            (allocation thunk))
          <))
  (list-ref runs 7))
