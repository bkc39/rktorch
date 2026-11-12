#lang racket/base

(module+ test
  (require (for-syntax racket/base syntax/parse)
           (only-in racket/dict dict-ref)
           (only-in racket/list append-map remove-duplicates shuffle)
           rackunit
           (only-in syntax/parse/define define-syntax-parse-rule)
           (only-in "../foreign/device-type.rkt" device)
           (only-in "../foreign/raw/accounts.rkt" account-of note-mark!)
           (only-in (submod "../foreign/raw/accounts.rkt" private) account-counters accounts)
           (only-in "../foreign/raw/cas.rkt" cas-update!)
           (only-in (submod "../foreign/raw/groups.rkt" private) group-shown)
           (only-in "../foreign/raw/memory.rkt"
                    native-memory-use/fold record-allocation! unaccount!)
           (only-in "../foreign/raw/tallies.rkt" flush-tallies! flush-tally!)
           (only-in (submod "../foreign/raw/tallies.rkt" private) own-group)
           (only-in "private/race-harness.rkt" spawn stop! with-race-hook)
           (only-in "private/spurious-cas.rkt" call-with-spurious-cas-failures)
           (only-in "private/within.rkt" bounded-test))

  (define mark (* 512 1024 1024))
  (define n 4096)

  (define (fold-of dev) (dict-ref (native-memory-use/fold) dev 0))
  (define (forming dev) (own-group (account-counters (account-of dev))))

  (define (call-with-fake-device index proc)
    (define dev (device 'cuda index))
    (dynamic-wind
     (lambda ()
       (note-mark! (account-of dev) mark)
       (flush-tallies!))
     (lambda () (proc dev))
     (lambda ()
       (flush-tallies!)
       (void (cas-update! accounts (lambda (all) (hash-remove all dev)))))))

  (define-syntax-parse-rule (with-batched-device (dev:id index:expr) body:expr ...+)
    (call-with-fake-device index (lambda (dev) body ...)))

  (define (shown-total groups)
    (for/sum ([g (in-list groups)]) (group-shown g)))

  (define (shown-deltas g thunk)
    (define deltas (box '()))
    (define (note! name subject)
      (when (and (eq? name 'group-shown) (eq? (car subject) g))
        (cas-update! deltas (lambda (ds) (cons (cdr subject) ds)))))
    (with-race-hook note! (thunk))
    (reverse (unbox deltas)))

  ;; The group a charge joins is its thread's forming group when the charge
  ;; starts, since only that thread and the test's own flushes close it.
  (define (churn! dev mine seen others rng)
    (define ring (make-vector 128 #f))
    (let loop ([i 0])
      (define key (box i))
      (define slot (modulo i 128))
      (define old (vector-ref ring slot))
      (set-box! seen (cons (forming dev) (unbox seen)))
      (set-box! mine (cons key (unbox mine)))
      (record-allocation! key (* 512 (add1 (modulo i 7))) dev)
      (vector-set! ring slot key)
      (when (and old (odd? i))
        (unaccount! old))
      (when (zero? (modulo i 5))
        (define theirs (unbox (vector-ref others (random (vector-length others) rng))))
        (unless (null? theirs)
          (unaccount! (car theirs))))
      (loop (add1 i))))

  (define (churn-and-kill! dev)
    (define rng (make-pseudo-random-generator))
    (define held (for/vector ([_ (in-range 8)]) (box '())))
    (define seen (for/list ([_ (in-range 8)]) (box '())))
    (define workers
      (for/list ([mine (in-vector held)] [groups (in-list seen)])
        (spawn (lambda () (churn! dev mine groups held (make-pseudo-random-generator)))
               #:pool 'own)))
    (for ([w (in-list (shuffle workers))])
      (sleep (* 0.02 (random rng)))
      (stop! w))
    (flush-tallies!)
    (values (append-map unbox (vector->list held))
            (remove-duplicates (append-map unbox seen) eq?)))

  (bounded-test "L14: eight churners killed at random leave the groups' phantoms at the fold"
    (with-batched-device (dev 60)
      (define-values (keys groups) (churn-and-kill! dev))
      (check-true (> (length groups) 8) "the churners closed groups")
      (check-equal? (shown-total groups) (fold-of dev) "phantom bytes against the fold")
      (for-each unaccount! keys)
      (flush-tallies!)
      (check-equal? (fold-of dev) 0)
      (check-equal? (shown-total groups) 0 "every group is dropped once its entries go")))

  (bounded-test "L14: releases lower their group once per flush, by their sum"
    (with-batched-device (dev 61)
      (flush-tally!)
      (define g (forming dev))
      (define keys (for/list ([i (in-range 32)]) (box i)))
      (for ([k (in-list keys)])
        (record-allocation! k n dev))
      (check-eq? (forming dev) g "32 small charges stay one forming batch")
      (check-equal? (group-shown g) 0 "a forming group has no phantom")
      (check-equal? (shown-deltas g flush-tally!) (list (* 32 n))
                    "the flush shows the batch in one phantom")
      (define deltas
        (shown-deltas g
                      (lambda ()
                        (for-each unaccount! keys)
                        (check-equal? (group-shown g) (* 32 n)
                                      "no release lowers the group before its flush")
                        (flush-tally!))))
      (check-equal? deltas (list (- (* 32 n))) "one lowering, by the releases' sum")
      (check-equal? (fold-of dev) 0)))

  ;; the charge region cannot park, so the hook runs the rival release and
  ;; the flush that closes the charge's group itself, once the entry is out
  (bounded-test "L14: a release that reaches a group before its charge's flush leaves it exact"
    (with-batched-device (dev 62)
      (define key (box 'raced))
      (flush-tally!)
      (define g (forming dev))
      (define (release-and-close name subject)
        (when (and (eq? name 'ledger-entry-added) (eq? subject key))
          (unaccount! key)
          (flush-tallies!)))
      (with-race-hook release-and-close
        (record-allocation! key n dev))
      (check-not-eq? (forming dev) g "the rival flush closed the charge's group")
      (flush-tallies!)
      (check-equal? (group-shown g) 0)
      (check-equal? (fold-of dev) 0)))

  (bounded-test "L14: spurious CAS failures at the groups lose nothing"
    (with-batched-device (dev 63)
      (define charged
        (for/list ([i (in-range 300)])
          (define key (box i))
          (define g (forming dev))
          (call-with-spurious-cas-failures
           '(group tally)
           (lambda ()
             (record-allocation! key 256 dev)
             (when (odd? i) (unaccount! key))
             (when (zero? (modulo i 7)) (flush-tally!))))
          (cons key g)))
      (define groups (remove-duplicates (map cdr charged) eq?))
      (flush-tallies!)
      (check-equal? (shown-total groups) (* 150 256))
      (check-equal? (fold-of dev) (* 150 256))
      (for-each unaccount! (map car charged))
      (flush-tallies!)
      (check-equal? (shown-total groups) 0))))
