#lang racket/base

(module+ test
  (require (only-in racket/list range)
           rackunit
           (only-in "../foreign.rkt" native-memory-limit)
           (only-in "../foreign/raw/pressure.rkt" collect-at-trough!)
           (only-in "private/race-harness.rkt"
                    check-ledger churn-once! env-number fuzz-hook hooks
                    join-all ledger-snapshot make-watch seeded-generator
                    settle! spawn watch-hook with-race-hook))

  (define mib (* 1024 1024))
  (define threads 4)
  (define rounds 12)

  (define seeds
    (cond
      [(env-number "RKTORCH_RACE_SEED" #f) => list]
      [else (range 1 (add1 (env-number "RKTORCH_RACE_SEEDS" 16)))]))

  (define (churn! seed label)
    (define rng (seeded-generator seed (list 'work label)))
    (for ([i (in-range rounds)])
      (churn-once! rng)
      (when (and (eqv? label 0) (zero? (modulo i 4)))
        (collect-at-trough!))))

  (define (scenario seed)
    (settle!)
    (define base (ledger-snapshot))
    (define w (make-watch))
    (with-race-hook (hooks (watch-hook w) (fuzz-hook seed))
      (parameterize ([native-memory-limit (* 4 mib)])
        (join-all
         (for/list ([k (in-range threads)])
           (spawn (lambda () (churn! seed k))
                  #:pool (and (even? k) 'own)
                  #:label k))))
      (settle!))
    (with-check-info (['seed seed]
                      ['replay (format "RKTORCH_RACE_SEED=~a raco test ~a"
                                       seed "torch/tests/race-fuzz-test.rkt")])
      (check-ledger base w)))

  (test-case "seeded yields at every race point keep the ledger invariants"
    (for ([seed (in-list seeds)])
      (scenario seed))))
