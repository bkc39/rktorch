#lang racket/base

(module+ test
  (require rackunit
           (only-in "../foreign/raw/accounts.rkt" account-of)
           (only-in (submod "../foreign/raw/accounts.rkt" private) account-counters)
           (only-in "../foreign/raw/memory.rkt" record-allocation! unaccount!)
           (only-in "../foreign/raw/phantoms.rkt" charge-epoch young-epoch?)
           (only-in (submod "../foreign/raw/phantoms.rkt" private) young-count-of)
           (only-in "../foreign/raw/tallies.rkt"
                    flush-tallies! flush-tally! pending-of tally-live!)
           (only-in "private/race-harness.rkt"
                    gate-hook make-gate release-gate! wait-for-arrival with-race-hook)
           (only-in "private/tally-devices.rkt" kib phantoms-of)
           (only-in "private/within.rkt" bounded-test)
           (only-in "private/young-epochs.rkt"
                    check-conserved unaccount-all! within-one-epoch))

  (define (own-pending dev)
    (pending-of (account-counters (account-of dev))))

  (define (stray-release-into-a-live-tag dev _on-other)
    (flush-tally!)
    (define p (own-pending dev))
    (define e1 (charge-epoch))
    (collect-garbage 'minor)
    (define key (box 'charged))
    (record-allocation! key (* 16 kib) dev)
    (define e2 (charge-epoch))
    (tally-live! p (- (* 8 kib)) -1 (- (* 8 kib)) e1)
    (flush-tally!)
    (define count (young-count-of (phantoms-of dev)))
    (tally-live! p (* 8 kib) 1 0)
    (unaccount-all! (list key))
    (check-conserved dev "after the releases")
    (and (young-epoch? e2) (not (young-epoch? e1)) count))

  (bounded-test "L14: a release of an ended epoch gathered into a live tag leaves its count"
    (check-equal? (within-one-epoch 130 stray-release-into-a-live-tag) (* 16 kib)
                  "the live epoch's young count after the stray release"))

  (define ((flush-at point #:collect? [collect? #f]) dev on-other)
    (define p (on-other (lambda () (flush-tally!) (own-pending dev))))
    (define released (box 'released))
    (define others (box 'others))
    (on-other (lambda ()
                (record-allocation! released (* 8 kib) dev)
                (flush-tally!)
                (unaccount! released)))
    (collect-garbage 'minor)
    (record-allocation! others (* 16 kib) dev)
    (flush-tally!)
    (define e1 (charge-epoch))
    (define g (make-gate point #:when (lambda (subject) (eq? subject p))))
    (define arrived
      (with-race-hook (gate-hook g)
        (define gathering
          (thread (lambda () (on-other (lambda () (tally-live! p (* 4 kib) 1 (* 4 kib) e1))))))
        (begin0 (wait-for-arrival g)
                (flush-tallies!)
                (when collect? (collect-garbage 'minor))
                (release-gate! g)
                (sync/timeout 30 gathering))))
    (define e2 (charge-epoch))
    (on-other flush-tally!)
    (define count (young-count-of (phantoms-of dev)))
    (on-other (lambda () (tally-live! p (- (* 4 kib)) -1 0)))
    (unaccount-all! (list others))
    (check-conserved dev "after the releases")
    (check-equal? arrived 'parked "the gather parked at the point")
    (and (young-epoch? e2)
         (eq? collect? (not (young-epoch? e1)))
         count))

  (bounded-test "L14: a flush between the clamp and the re-tag keeps the young count"
    (check-equal? (within-one-epoch 140 (flush-at 'young-clamped)) (* 20 kib)
                  "this epoch's young count: the other charge and this one"))

  (bounded-test "L14: a flush between the re-tag and the add keeps the young count"
    (check-equal? (within-one-epoch 150 (flush-at 'young-retagged)) (* 20 kib)
                  "this epoch's young count: the other charge and this one"))

  (bounded-test "L14: a collection after a flush between the re-tag and the add keeps the charge young"
    (check-equal? (within-one-epoch 160 (flush-at 'young-retagged #:collect? #t)) (* 4 kib)
                  "the next epoch's young count: this charge")))
