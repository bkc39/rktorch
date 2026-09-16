#lang racket/base

(module+ test
  (require rackunit
           (only-in "../foreign.rkt"
                    cpu-device finalizer-diagnostics native-memory-limit
                    native-memory-use reclaim-native-memory! zeros)
           (only-in "../foreign/raw/memory.rkt" native-memory-use/fold))

  (define mib (* 1024 1024))

  (define (cpu-bytes)
    (cond [(assoc (cpu-device) (native-memory-use)) => cdr]
          [else 0]))

  (define (collections)
    (cdr (assq 'pressure-collections (finalizer-diagnostics))))

  (define (settle!)
    (for ([_ (in-range 3)])
      (reclaim-native-memory!)))

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
                  (format "~a collections for 120 MiB of live growth" fired)))))
