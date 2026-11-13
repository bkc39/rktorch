#lang racket/base

(require (only-in racket/dict dict-ref)
         (only-in rackunit check-equal?)
         (only-in "../../foreign/raw/memory.rkt" native-memory-use/fold unaccount!)
         (only-in "../../foreign/raw/tallies.rkt" flush-tallies!)
         (only-in "tally-devices.rkt" charged shared-live with-batched-device))

(provide check-conserved
         unaccount-all!
         within-one-epoch)

(define (fold-of dev) (dict-ref (native-memory-use/fold) dev 0))

(define (unaccount-all! keys)
  (for-each unaccount! keys)
  (flush-tallies!))

(define (check-conserved dev what)
  (define live (shared-live dev))
  (check-equal? live (fold-of dev) (format "~a: shared live against the fold" what))
  (check-equal? (charged dev) (max 0 live) (format "~a: the phantoms against live" what)))

(define (call-with-other-thread proc)
  (define asks (make-channel))
  (define answers (make-channel))
  (define other
    (thread (lambda ()
              (let loop ()
                (channel-put answers ((channel-get asks)))
                (loop)))))
  (define (on-other thunk)
    (channel-put asks thunk)
    (channel-get answers))
  (begin0 (proc on-other)
          (kill-thread other)))

(define (within-one-epoch first scenario)
  (for/or ([index (in-range first (+ first 10))])
    (with-batched-device (dev index)
      (call-with-other-thread (lambda (on-other) (scenario dev on-other))))))
