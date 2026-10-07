#lang racket/base

(module+ test
  (require rackunit
           (only-in "../foreign.rkt"
                    cpu-device finalizer-diagnostics native-memory-use
                    reclaim-native-memory! zeros)
           (only-in (submod "../foreign.rkt" unsafe) tensor-free!)
           (only-in "../foreign/raw/device-queries.rkt" install-device-queries!)
           (only-in "../foreign/raw/pressure.rkt"
                    backstop-interval collect-at-trough! native-collect-margin native-memory-limit release-spacing
                    reset-pressure-state!))

  (define mib (* 1024 1024))
  (define block (* 4 mib))

  (define (collections)
    (cdr (assq 'pressure-collections (finalizer-diagnostics))))

  (define (interval)
    (backstop-interval (cpu-device)))

  (define (cpu-bytes)
    (cond [(assoc (cpu-device) (native-memory-use)) => cdr]
          [else 0]))

  ;; The stand-in allocator reads what the ledger held when the test began
  ;; plus a cache, and the mark sits 512 MiB above that. A release empties
  ;; the cache, which by the next reading holds `comes-back` again: a step's
  ;; working set taking its bytes straight back, or nothing for a release
  ;; that stays released.
  (define held 0)
  (define cache 0)
  (define comes-back 0)
  (define mark 0)

  (define (install-stand-in! #:empties? empties?)
    (install-device-queries!
     #:capacity (lambda (_dev) #f)
     #:allocated (lambda (_dev)
                   (begin0 (+ held cache) (set! cache (max cache comes-back))))
     #:release (lambda (_dev)
                 (when empties? (set! cache 0))
                 #t)))

  (define (settle!)
    (install-device-queries! #:capacity (lambda (_dev) #f)
                             #:allocated (lambda (_dev) #f))
    (for ([_ (in-range 3)])
      (reclaim-native-memory!))
    (reset-pressure-state!)
    (set! held (cpu-bytes))
    (set! mark (+ held (* 512 mib))))

  ;; in 4 MiB tensors, each freed at once, so the ledger never moves the
  ;; reading off the stand-in's
  (define (allocate! share-of-mark #:spacing [spacing 0])
    (parameterize ([native-memory-limit mark]
                   [release-spacing spacing])
      (for ([_ (in-range (ceiling (/ (* share-of-mark mark) block)))])
        (tensor-free! (zeros 1024 1024)))))

  ;; a step's end that reads the allocator but has nothing to collect
  (define (trough!)
    (parameterize ([native-memory-limit mark]
                   [native-collect-margin (* 1024 1024 mib)])
      (collect-at-trough!)))

  (define a-minute 60000.0)

  (test-case "a release whose bytes come back backs the backstop off"
    (settle!)
    (install-stand-in! #:empties? #t)
    (set! cache (* 768 mib))
    (set! comes-back (* 768 mib))
    (define before (collections))
    (allocate! 39/10)
    (define fired (- (collections) before))
    ;; an interval that doubles from an eighth of the mark fires after 1/8,
    ;; 1/4, 1/2, 1 and 2 marks; one reset by every release fires 31 times
    (check-true (<= 3 fired 8)
                (format "~a backstop collections over 3.9 marks" fired))
    (check-equal? (interval) (* 2 mark)
                  "every release came straight back, so the interval must back off")
    (settle!))

  (test-case "a release that stays released earns the reset"
    (settle!)
    (install-stand-in! #:empties? #f)
    (set! cache (* 768 mib))
    (set! comes-back 0)
    ;; releases that free nothing back off, firing after 1/8, 3/8, 7/8 and
    ;; 15/8 marks
    (allocate! 2)
    (check-equal? (interval) (* 2 mark))
    (install-stand-in! #:empties? #t)
    (define before (collections))
    (allocate! 2)
    (check-equal? (- (collections) before) 1
                  "the reopened gate should fire once, with a release that holds")
    (check-equal? (interval) (* 2 mark)
                  "a release has to stay released before it earns the reset")
    (allocate! 17/8)
    (check-equal? (interval) (quotient mark 8)
                  "a release that stayed released for a mark's worth kept the back-off")
    (check-equal? (- (collections) before) 1
                  "the backstop fired again though the cache stayed empty")
    ;; later pressure finds the interval at the base, and a release that
    ;; stays released keeps it there
    (set! cache (* 768 mib))
    (allocate! 1/2)
    (check-equal? (- (collections) before) 2 "the new pressure went unnoticed")
    (check-equal? (interval) (quotient mark 8) "a release on trial moved the interval")
    (allocate! 2)
    (check-equal? (- (collections) before) 2
                  "the backstop fired again though the cache stayed empty")
    (check-equal? (interval) (quotient mark 8) "a release that held moved the interval")
    (settle!))

  ;; the cache comes back after the trial's mark of allocation, but while the
  ;; spacing still holds the next release back
  (test-case "a release stays on trial until its spacing has passed"
    (settle!)
    (install-stand-in! #:empties? #t)
    (set! cache (* 768 mib))
    (set! comes-back 0)
    (allocate! 1/4 #:spacing a-minute)
    (allocate! 2 #:spacing a-minute)
    (check-equal? (interval) (quotient mark 8))
    (set! cache (* 768 mib))
    (define before (collections))
    (allocate! 2 #:spacing a-minute)
    (define fired (- (collections) before))
    (check-equal? (interval) (* 2 mark)
                  "a firing inside the spacing must back off, not stay at the base")
    (check-true (<= 1 fired 6)
                (format "~a backstop collections over 2 marks inside the spacing"
                        fired))
    (settle!))

  ;; fruitless releases first raise the interval past what is allocated
  ;; between the troughs below, so the gate never reopens to end the trial
  (test-case "a trough credits a trial the gate never reopened to see out"
    (settle!)
    (install-stand-in! #:empties? #f)
    (set! cache (* 768 mib))
    (set! comes-back 0)
    (allocate! 2)
    (install-stand-in! #:empties? #t)
    (allocate! 2)
    (check-equal? (interval) (* 2 mark) "the release went on trial at the cap")
    (allocate! 3/2)
    (set! cache (* 768 mib))
    (trough!)
    (check-equal? (interval) (* 2 mark)
                  "a trough reading over the mark must not credit the release")
    (set! cache 0)
    (trough!)
    (check-equal? (interval) (quotient mark 8)
                  "a trough under the mark after the trial must credit it")
    (settle!)))
