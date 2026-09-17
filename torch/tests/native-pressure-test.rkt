#lang racket/base

(module+ test
  (require rackunit
           (only-in "../foreign.rkt"
                    cpu-device finalizer-diagnostics native-memory-limit
                    native-memory-use reclaim-native-memory! with-no-grad
                    zeros)
           (only-in "../foreign/raw/memory.rkt" native-memory-use/fold)
           (only-in "../foreign/raw/pressure.rkt"
                    allocator-reading call-as-the-collector collect-at-trough!
                    reset-pressure-state! trough-budget trough-margin)
           (only-in "../nn.rkt" Linear Sequential))

  (define mib (* 1024 1024))

  (define (cpu-bytes)
    (cond [(assoc (cpu-device) (native-memory-use)) => cdr]
          [else 0]))

  (define (collections)
    (cdr (assq 'pressure-collections (finalizer-diagnostics))))

  (define (trough-minors)
    (cdr (assq 'trough-minors (finalizer-diagnostics))))

  (define (trough-collections)
    (cdr (assq 'trough-collections (finalizer-diagnostics))))

  (define (settle!)
    (for ([_ (in-range 3)])
      (reclaim-native-memory!))
    (reset-pressure-state!))

  ;; a step: k temporaries of 4 MiB live together, then drop as a batch
  (define (step! k)
    (length (for/list ([_ (in-range k)]) (zeros 1024 1024))))

  (define (positive-entries totals)
    (filter (lambda (entry) (positive? (cdr entry))) totals))

  (test-case "the running totals agree with the entry-by-entry fold"
    (settle!)
    (define held (for/list ([_ (in-range 5)]) (zeros 256 256)))
    (check-equal? (positive-entries (native-memory-use))
                  (positive-entries (native-memory-use/fold)))
    (check-equal? (length held) 5))

  (test-case "no limit and no known capacity: the trigger never fires"
    (settle!)
    (define before (collections))
    (for ([_ (in-range 8)])
      (step! 8))
    (check-equal? (collections) before))

  (test-case "under a limit the ledger stays bounded and the trigger counts"
    (settle!)
    (define base (cpu-bytes))
    (define before (collections))
    (parameterize ([native-memory-limit (* 64 mib)])
      (define high-water
        (for/fold ([hw 0]) ([_ (in-range 20)])
          (step! 8)
          (max hw (- (cpu-bytes) base))))
      (check-true (> (collections) before) "the trigger never fired")
      (check-true (< high-water (* 128 mib))
                  (format "ledger reached ~a MiB under a 64 MiB limit"
                          (quotient high-water mib)))))

  (test-case "a working set above the mark backs off instead of thrashing"
    (settle!)
    (define before (collections))
    (parameterize ([native-memory-limit (* 64 mib)])
      (define resident (for/list ([_ (in-range 20)]) (zeros 1024 1024)))
      (define kept (for/list ([_ (in-range 120)]) (zeros 512 512)))
      (check-equal? (+ (length resident) (length kept)) 140)
      (define fired (- (collections) before))
      (check-true (> fired 0) "the trigger never fired")
      (check-true (<= fired 5)
                  (format "~a collections for 120 MiB of live growth" fired))))

  (test-case "an allocator reading above the mark fires with a small ledger"
    (settle!)
    (define before (collections))
    (parameterize ([native-memory-limit (* 64 mib)]
                   [allocator-reading (lambda (_dev) (* 200 mib))])
      (define kept (for/list ([_ (in-range 48)]) (zeros 512 512)))
      (check-equal? (length kept) 48)
      (check-true (< (cpu-bytes) (* 64 mib)) "the ledger alone is under the mark")
      (define fired (- (collections) before))
      (check-true (> fired 0) "the allocator reading never fired the trigger")
      (check-true (<= fired 4)
                  (format "~a collections against a reading that cannot fall"
                          fired))))

  (test-case "an unknown allocator reading leaves the ledger in charge"
    (settle!)
    (define before (collections))
    (parameterize ([native-memory-limit (* 64 mib)]
                   [allocator-reading (lambda (_dev) #f)])
      (define kept (for/list ([_ (in-range 48)]) (zeros 512 512)))
      (check-equal? (length kept) 48)
      (check-equal? (collections) before)))

  (test-case "residue past the margin at a trough is collected"
    (settle!)
    (collect-at-trough!)
    (define base (cpu-bytes))
    (define before (trough-collections))
    (parameterize ([trough-margin (* 16 mib)])
      (step! 8)
      (collect-at-trough!)
      (check-equal? (- (trough-collections) before) 1)
      (check-true (< (- (cpu-bytes) base) (* 16 mib))
                  "the trough collection left the step's residue behind")))

  (test-case "residue under the margin is left alone"
    (settle!)
    (collect-at-trough!)
    (define before (trough-collections))
    (parameterize ([trough-margin (* 64 mib)])
      (define kept (step! 4))
      (collect-at-trough!)
      (check-equal? kept 4)
      (check-equal? (trough-collections) before)))

  (test-case "the budget spaces trough collections out"
    (settle!)
    (collect-at-trough!)
    (define before (trough-collections))
    (parameterize ([trough-margin (* 16 mib)]
                   [trough-budget 1/1000])
      (for ([_ (in-range 5)])
        (step! 8)
        (collect-at-trough!))
      (check-equal? (- (trough-collections) before) 1)))

  (test-case "an outermost layer call with gradients off is a trough, once"
    (settle!)
    (define net (Sequential (Linear 1024 1024) (Linear 1024 1024)))
    (define x (zeros 1024 1024))
    (parameterize ([trough-margin (* 1 mib)]
                   [trough-budget 1000])
      (define before (trough-minors))
      (define y (with-no-grad (net x)))
      (check-equal? (- (trough-minors) before) 1
                    "the nested Linear calls must not count")
      (check-true (and y #t))))

  (test-case "with gradients on a layer call is the peak, not a trough"
    (settle!)
    (define net (Linear 1024 1024))
    (define x (zeros 1024 1024))
    (parameterize ([trough-margin (* 1 mib)]
                   [trough-budget 1000])
      (define before (trough-minors))
      (define y (net x))
      (check-equal? (trough-minors) before)
      (check-true (and y #t))))

  (test-case "a collector killed mid-collection does not hold the claim"
    (settle!)
    (define inside (make-semaphore 0))
    (define doomed
      (thread (lambda ()
                (call-as-the-collector
                 (lambda ()
                   (semaphore-post inside)
                   (sync never-evt))))))
    (semaphore-wait inside)
    (define skipped? #t)
    (call-as-the-collector (lambda () (set! skipped? #f)))
    (check-true skipped? "a live claimant must make a second collector skip")
    (kill-thread doomed)
    (define before (trough-collections))
    (parameterize ([trough-margin (* 16 mib)])
      (step! 8)
      (collect-at-trough!)
      (check-equal? (- (trough-collections) before) 1)))

  (test-case "the diagnostics carry every collection counter"
    (define keys (map car (finalizer-diagnostics)))
    (for ([k (in-list '(runs failures messages ledger-entries
                        pressure-collections pressure-reclaimed
                        trough-collections trough-minors))])
      (check-not-false (memq k keys) (format "missing ~a" k)))))
