#lang racket/base

(module+ test
  (require rackunit
           (only-in "../foreign.rkt"
                    add backward! cpu-device finalizer-diagnostics grad matmul
                    native-collect-at-troughs native-memory-limit
                    native-memory-unaccounted native-memory-use
                    mps-empty-cache! reclaim-native-memory! sum with-no-grad
                    zeros)
           (only-in "../foreign/raw/memory.rkt"
                    collect-and-drain! native-memory-use/fold)
           (only-in "../foreign/raw/collector.rkt"
                    call-as-the-collector collect-and-wait! drain-deadline)
           (only-in "../foreign/raw/device-queries.rkt"
                    allocator-reading install-device-queries!)
           (only-in "../foreign/raw/pressure.rkt"
                    collect-at-trough! margin-over
                    native-collect-budget native-collect-margin
                    native-memory-fraction release-spacing
                    reset-pressure-state! shadow-refresh lower-shadows!)
           (only-in "../nn.rkt"
                    Linear Sequential gen:layer))

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

  (define (trough-floor)
    (cdr (assq 'trough-floor (finalizer-diagnostics))))

  (define no-backoff +inf.0)

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

  (define (cpu-unaccounted)
    (cond [(assoc (cpu-device) (native-memory-unaccounted)) => cdr]
          [else 0]))

  (define idle-margin (* 1024 1024 mib))

  (test-case "a trough charges what only the allocator holds, as phantom bytes"
    (settle!)
    (define held (for/list ([_ (in-range 4)]) (zeros 1024 1024)))
    (define before (current-memory-use))
    (parameterize ([allocator-reading (lambda (_dev) (+ (cpu-bytes) (* 256 mib)))]
                   [native-collect-margin idle-margin])
      (collect-at-trough!))
    (check-true (<= (* 255 mib) (cpu-unaccounted) (* 256 mib))
                (format "~a MiB unaccounted" (quotient (cpu-unaccounted) mib)))
    (check-true (>= (- (current-memory-use) before) (* 200 mib))
                "the collector was not charged for the allocator's excess")
    (parameterize ([native-collect-margin idle-margin])
      (collect-at-trough!))
    (check-true (>= (cpu-unaccounted) (* 255 mib))
                "an allocator that cannot answer must leave the charge as it was")
    (parameterize ([allocator-reading (lambda (_dev) (cpu-bytes))]
                   [native-collect-margin idle-margin])
      (collect-at-trough!))
    (check-equal? (cpu-unaccounted) 0
                  "an allocator holding only the ledger's bytes must clear the charge")
    (check-equal? (length held) 4))

  (test-case "reclaiming clears a charge the allocator no longer backs"
    (settle!)
    (parameterize ([allocator-reading (lambda (_dev) (+ (cpu-bytes) (* 256 mib)))]
                   [native-collect-margin idle-margin])
      (collect-at-trough!))
    (check-true (positive? (cpu-unaccounted)))
    (parameterize ([allocator-reading (lambda (_dev) (cpu-bytes))])
      (reclaim-native-memory!))
    (check-equal? (cpu-unaccounted) 0
                  "the charge outlived the reclamation that ended it"))

  ;; the stand-in's hidden storage lives until the trough's own collection
  ;; has run, as a dropped graph's would until its owner is collected
  (define (reclaimed) (cdr (assq 'pressure-reclaimed (finalizer-diagnostics))))

  (test-case "a trough whose collection frees hidden storage stops charging it"
    (settle!)
    (define before (trough-collections))
    (define reclaimed-before (reclaimed))
    (define (hidden) (if (= (trough-collections) before) (* 256 mib) 0))
    (parameterize ([allocator-reading (lambda (_dev) (+ (cpu-bytes) (hidden)))]
                   [native-collect-margin (* 16 mib)]
                   [native-collect-budget no-backoff])
      (step! 8)
      (collect-at-trough!))
    (check-equal? (- (trough-collections) before) 1 "the trough did not collect")
    (check-equal? (cpu-unaccounted) 0
                  "the charge outlived the storage the collection freed")
    (check-true (>= (- (reclaimed) reclaimed-before) (* 256 mib))
                "the freed hidden storage is missing from pressure-reclaimed"))

  ;; a zero deadline makes the trough's drain report that it ran out of time
  (test-case "a trough whose drain stalls leaves the charge for the next one"
    (settle!)
    (define before (trough-collections))
    (define (hidden) (if (= (trough-collections) before) (* 256 mib) 0))
    (parameterize ([allocator-reading (lambda (_dev) (+ (cpu-bytes) (hidden)))]
                   [native-collect-margin (* 16 mib)]
                   [native-collect-budget no-backoff]
                   [drain-deadline 0])
      (step! 8)
      (collect-at-trough!))
    (check-equal? (- (trough-collections) before) 1 "the trough did not collect")
    (check-true (>= (cpu-unaccounted) (* 255 mib))
                "a stalled drain refreshed from finalizers still to run")
    (settle!))

  ;; a stand-in reading that holds steady, as the allocator's does when the
  ;; backward pass has written a gradient no handle points at yet
  (test-case "wrapping a gradient moves its bytes off the shadow"
    (settle!)
    (define w (zeros 1024 1024 #:requires-grad? #t))
    (parameterize ([allocator-reading (lambda (_dev) (+ (cpu-bytes) (* 256 mib)))]
                   [native-collect-margin idle-margin])
      (backward! (sum (matmul w w)))
      (define charged (cpu-unaccounted))
      (define g (grad w))
      (check-equal? (- charged (cpu-unaccounted)) (* 4 mib)
                    "the gradient is charged as a tensor and in the shadow")
      ;; clipping and then the optimizer each take the gradient
      (define g-again (grad w))
      (check-equal? (- charged (cpu-unaccounted)) (* 4 mib)
                    "a second handle on the same gradient moved its bytes again")
      ;; the next step's refresh charges the gradient anew
      (backward! (sum (matmul w w)))
      (define recharged (cpu-unaccounted))
      (define g-next (grad w))
      (check-equal? (- recharged (cpu-unaccounted)) (* 4 mib)
                    "after a new refresh the gradient's bytes did not move")
      (check-true (and g g-again g-next #t)))
    (settle!))

  ;; the optimizer's handle on a gradient dies after the step; the parameter
  ;; still holds the storage
  (test-case "a gradient's handle gives its bytes back to the shadow when it dies"
    (settle!)
    (define w (zeros 1024 1024 #:requires-grad? #t))
    (parameterize ([allocator-reading (lambda (_dev) (+ (cpu-bytes) (* 256 mib)))]
                   [native-collect-margin idle-margin])
      (backward! (sum (matmul w w))))
    (define charged (cpu-unaccounted))
    (define (take-and-drop!) (void (grad w)))
    (take-and-drop!)
    (check-equal? (- charged (cpu-unaccounted)) (* 4 mib) "the gradient was not adopted")
    (collect-and-wait!)
    (check-equal? (cpu-unaccounted) charged
                  "the dead handle's bytes did not go back to the shadow")
    (check-true (and w #t))
    (settle!))

  ;; with no charge to move (no allocator figure, as on the CPU), there is
  ;; nothing for the release to give back
  (test-case "a gradient the shadow never charged gives nothing back"
    (settle!)
    (define w (zeros 1024 1024 #:requires-grad? #t))
    (parameterize ([native-collect-margin idle-margin])
      (backward! (sum (matmul w w))))
    (define (take-and-drop!) (void (grad w)))
    (take-and-drop!)
    (collect-and-wait!)
    (check-equal? (cpu-unaccounted) 0
                  "a release gave back bytes its adoption never took")
    (check-true (and w #t))
    (settle!))

  (test-case "the charge never counts a ledger byte twice"
    (settle!)
    (define held (for/list ([_ (in-range 4)]) (zeros 1024 1024)))
    (parameterize ([allocator-reading (lambda (_dev) (quotient (cpu-bytes) 2))]
                   [native-collect-margin idle-margin])
      (collect-at-trough!))
    (check-equal? (cpu-unaccounted) 0)
    (check-equal? (length held) 4))

  (test-case "with the shadow off a trough charges nothing"
    (settle!)
    (parameterize ([shadow-refresh #f]
                   [allocator-reading (lambda (_dev) (* 256 mib))]
                   [native-collect-margin idle-margin])
      (collect-at-trough!))
    (check-equal? (cpu-unaccounted) 0))

  (test-case "'samples also refreshes the charge at the backstop's samples"
    (settle!)
    (parameterize ([shadow-refresh 'samples]
                   [native-memory-limit (* 64 mib)]
                   [allocator-reading (lambda (_dev) (* 200 mib))])
      (define kept (for/list ([_ (in-range 48)]) (zeros 512 512)))
      (check-equal? (length kept) 48)
      (check-true (positive? (cpu-unaccounted))
                  "no trough ran, so only a sample could have set it"))
    (define charged (cpu-unaccounted))
    (parameterize ([shadow-refresh 'samples]
                   [native-memory-limit (* 64 mib)]
                   [allocator-reading (lambda (_dev) #f)])
      (define more (for/list ([_ (in-range 48)]) (zeros 512 512)))
      (check-equal? (length more) 48))
    (check-equal? (cpu-unaccounted) charged
                  "a sample that could not read the allocator changed the charge")
    (settle!)
    (check-equal? (cpu-unaccounted) 0 "a reset must clear the charge"))

  (test-case "residue past the margin at a trough is collected"
    (settle!)
    (collect-at-trough!)
    (define base (cpu-bytes))
    (define before (trough-collections))
    (parameterize ([native-collect-margin (* 16 mib)])
      (step! 8)
      (collect-at-trough!)
      (check-equal? (- (trough-collections) before) 1)
      (check-true (< (- (cpu-bytes) base) (* 16 mib))
                  "the trough collection left the step's residue behind")))

  (test-case "residue under the margin is left alone"
    (settle!)
    (collect-at-trough!)
    (define before (trough-collections))
    (parameterize ([native-collect-margin (* 64 mib)])
      (define kept (step! 4))
      (collect-at-trough!)
      (check-equal? kept 4)
      (check-equal? (trough-collections) before)))

  (test-case "the budget spaces trough collections out"
    (settle!)
    (collect-at-trough!)
    (define before (trough-collections))
    (parameterize ([native-collect-margin (* 16 mib)]
                   [native-collect-budget 1/1000])
      (for ([_ (in-range 5)])
        (step! 8)
        (collect-at-trough!))
      (check-equal? (- (trough-collections) before) 1)))

  (test-case "an outermost layer call with gradients off is a trough, once"
    (settle!)
    (define net (Sequential (Linear 1024 1024) (Linear 1024 1024)))
    (define x (zeros 1024 1024))
    (parameterize ([native-collect-margin (* 1 mib)]
                   [native-collect-budget 1000])
      (define before (trough-minors))
      (define y (with-no-grad (net x)))
      (check-equal? (- (trough-minors) before) 1
                    "the nested Linear calls must not count")
      (check-true (and y #t))))

  (test-case "with gradients on a layer call is the peak, not a trough"
    (settle!)
    (define net (Linear 1024 1024))
    (define x (zeros 1024 1024))
    (parameterize ([native-collect-margin (* 1 mib)]
                   [native-collect-budget 1000])
      (define before (trough-minors))
      (define y (net x))
      (check-equal? (trough-minors) before)
      (check-true (and y #t))))

  (test-case "applying a hand-written layer with gradients off is a trough, once"
    (struct Hand (w)
      #:methods gen:layer
      [(define (layer-forward self . inputs)
         (add (matmul (car inputs) (Hand-w self)) 1))])
    (define hand (Hand (zeros 1024 1024)))
    (define x (zeros 1024 1024))
    (parameterize ([native-collect-margin (* 1 mib)]
                   [native-collect-budget 1000])
      (settle!)
      (define before (trough-minors))
      (check-true (and (with-no-grad (hand x)) #t))
      (check-equal? (- (trough-minors) before) 1
                    "the application's return collects once")))

  (test-case "native-collect-at-troughs #f turns off both implicit collections"
    (define net (Linear 1024 1024))
    (define x (zeros 1024 1024))
    (define (train-step!) (backward! (sum (net x))))
    (parameterize ([native-collect-margin (* 1 mib)]
                   [native-collect-budget 1000])
      (settle!)
      (define on (trough-collections))
      (train-step!)
      (check-equal? (- (trough-collections) on) 1 "on by default")
      (parameterize ([native-collect-at-troughs #f])
        (settle!)
        (define full (trough-collections))
        (define minors (trough-minors))
        (train-step!)
        (check-true (and (with-no-grad (net x)) #t))
        (check-equal? (trough-collections) full "backward! did not collect")
        (check-equal? (trough-minors) minors "the layer call did not collect"))))

  (test-case "with the collections off, both troughs still refresh the charge"
    (define net (Linear 1024 1024))
    (define x (zeros 1024 1024))
    (define (charged-by thunk)
      (settle!)
      (parameterize ([native-collect-at-troughs #f]
                     [allocator-reading
                      (lambda (_dev) (+ (cpu-bytes) (* 256 mib)))])
        (thunk))
      (cpu-unaccounted))
    (check-true (positive? (charged-by (lambda () (backward! (sum (net x))))))
                "backward! left the charge where it was")
    (check-true (positive? (charged-by (lambda () (with-no-grad (net x)))))
                "the no-grad layer call left the charge where it was")
    (settle!))

  (test-case "the default margin is the floor, kept between 256 MiB and 1 GiB"
    (check-equal? (margin-over 0) (* 256 mib))
    (check-equal? (margin-over (* 100 mib)) (* 256 mib))
    (check-equal? (margin-over (* 512 mib)) (* 512 mib))
    (check-equal? (margin-over (* 4096 mib)) (* 1024 mib))
    (parameterize ([native-collect-margin (* 16 mib)])
      (check-equal? (margin-over (* 512 mib)) (* 16 mib))))

  ;; the tensors stay held, so each step tests the decision alone: whether a
  ;; collection reclaims anything is the other tests' business
  (test-case "with the default margin, troughs follow the floor"
    (settle!)
    (define (hold k) (for/list ([_ (in-range k)]) (zeros 1024 1024)))
    (define (collects? thunk)
      (define before (trough-collections))
      (thunk)
      (collect-at-trough!)
      (positive? (- (trough-collections) before)))
    (parameterize ([native-collect-budget no-backoff])
      (define held '())
      (define (grow! k) (set! held (cons (hold k) held)))
      (check-false (collects? (lambda () (grow! 32)))
                   "128 MiB is under the 256 MiB minimum")
      (check-true (collects? (lambda () (grow! 48)))
                  "320 MiB over an empty floor is past it")
      (check-false (collects? (lambda () (grow! 70)))
                   "280 MiB over a 320 MiB floor is under a margin that follows it")
      (check-true (collects? (lambda () (grow! 20)))
                  "360 MiB over a 320 MiB floor is past it")
      (check-equal? (length held) 4)))

  (test-case "a trough that does not collect still lowers the floor"
    (settle!)
    (define settled
      (parameterize ([native-collect-margin (* 16 mib)]
                     [native-collect-budget no-backoff])
        (define kept (for/list ([_ (in-range 40)]) (zeros 1024 1024)))
        (collect-at-trough!)
        (begin0 (trough-floor) (check-equal? (length kept) 40))))
    (check-true (>= settled (* 128 mib))
                (format "the floor settled at ~a MiB" (quotient settled mib)))
    (for ([_ (in-range 3)]) (reclaim-native-memory!))
    (define before (trough-collections))
    (parameterize ([native-collect-margin (* 1024 1024 mib)]
                   [native-collect-budget no-backoff])
      (collect-at-trough!))
    (check-equal? (- (trough-collections) before) 0
                  "a margin nothing can exceed must leave the trough idle")
    (check-true (< (trough-floor) settled)
                "the floor follows the ledger down with no collection at all"))

  ;; a zero deadline makes every drain report that it ran out of time
  (test-case "a stalled drain does not credit the backstop's byte gate"
    (settle!)
    (parameterize ([native-memory-limit (* 64 mib)]
                   [drain-deadline 0])
      ;; a working set over the mark, so every check that looks does collect
      (define held (for/list ([_ (in-range 20)]) (zeros 1024 1024)))
      (define before (collections))
      (define more (for/list ([_ (in-range 3)]) (zeros 1024 1024)))
      (check-equal? (+ (length held) (length more)) 23)
      (check-true (>= (- (collections) before) 3)
                  (format "~a collections over 3 allocations past the mark"
                          (- (collections) before)))))

  (test-case "a stalled drain at a trough leaves the floor where it was"
    (settle!)
    (collect-at-trough!)
    (define settled (trough-floor))
    (define held
      (parameterize ([native-collect-margin (* 16 mib)]
                     [native-collect-budget no-backoff]
                     [drain-deadline 0])
        (define before-stall (trough-collections))
        (define kept (for/list ([_ (in-range 8)]) (zeros 1024 1024)))
        (collect-at-trough!)
        (check-equal? (- (trough-collections) before-stall) 1)
        (check-equal? (trough-floor) settled
                      "a drain that ran out of time must not settle the floor")
        kept))
    ;; the floor never took the stalled collection's snapshot, so the same
    ;; residue is still over it at the next trough
    (parameterize ([native-collect-margin (* 16 mib)]
                   [native-collect-budget no-backoff])
      (define before-retry (trough-collections))
      (collect-at-trough!)
      (check-equal? (length held) 8)
      (check-equal? (- (trough-collections) before-retry) 1
                    "the stalled trough must not have settled the floor")))

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
    (parameterize ([native-collect-margin (* 16 mib)])
      (step! 8)
      (collect-at-trough!)
      (check-equal? (- (trough-collections) before) 1)))

  (test-case "the diagnostics carry every collection counter"
    (define keys (map car (finalizer-diagnostics)))
    (for ([k (in-list '(runs failures messages ledger-entries
                        pressure-collections pressure-reclaimed
                        trough-collections trough-minors trough-floor))])
      (check-not-false (memq k keys) (format "missing ~a" k))))

  ;; the CPU has no capacity of its own, so the fraction is exercised against
  ;; a stand-in; the real queries are restored at the end
  (test-case "the memory fraction scales the capacity-derived mark"
    (define (collections-over fraction blocks)
      (settle!)
      (define before (collections))
      (parameterize ([native-memory-fraction fraction])
        (define held (for/list ([_ (in-range blocks)]) (zeros 1024 1024)))
        (begin0 (- (collections) before)
                (check-equal? (length held) blocks))))
    (install-device-queries! #:capacity (lambda (_dev) (* 128 mib))
                             #:allocated (lambda (_dev) #f))
    ;; half of 128 MiB is a 64 MiB mark, and 20 blocks of 4 MiB pass it
    (check-true (positive? (collections-over 1/2 20)))
    ;; all of it is a 128 MiB mark, which the same 80 MiB stays under
    (check-equal? (collections-over 1 20) 0)
    ;; unset, the device type's share applies: 4/5 off MPS, a 102 MiB mark
    ;; that 80 MiB stays under and 120 MiB passes
    (check-equal? (collections-over #f 20) 0)
    (check-true (positive? (collections-over #f 30)))
    (install-device-queries! #:capacity (lambda (_dev) #f)
                             #:allocated (lambda (_dev) #f))
    (settle!))

  ;; a stand-in for a device whose cache the backstop can empty, as on MPS
  (test-case "a backstop empties a releasable cache, no more often than the spacing"
    (define releases 0)
    (define (fired-and-released spacing [deadline (drain-deadline)])
      (settle!)
      (set! releases 0)
      (define before (collections))
      (parameterize ([native-memory-fraction 1/2]
                     [release-spacing spacing]
                     [drain-deadline deadline])
        (define held (for/list ([_ (in-range 40)]) (zeros 1024 1024)))
        (check-equal? (length held) 40))
      (values (- (collections) before) releases))
    (install-device-queries! #:capacity (lambda (_dev) (* 128 mib))
                             #:allocated (lambda (_dev) #f)
                             #:release (lambda (_dev)
                                         (set! releases (add1 releases))
                                         #t))
    (define-values (fired spaced) (fired-and-released +inf.0))
    (check-true (>= fired 2) (format "~a backstop collections" fired))
    (check-equal? spaced 1 "the spacing must hold every later firing back")
    (define-values (fired-unspaced unspaced) (fired-and-released 0))
    (check-equal? unspaced fired-unspaced
                  "with no spacing every firing must release")
    ;; a zero deadline makes every drain report that it ran out of time
    (define-values (fired-stalled stalled) (fired-and-released 0 0))
    (check-true (positive? fired-stalled))
    (check-equal? stalled 0
                  "a drain that ran out of time must not release the cache")
    (install-device-queries! #:capacity (lambda (_dev) #f)
                             #:allocated (lambda (_dev) #f))
    (settle!))

  ;; the stand-in's allocator holds a cache no release frees, so a release
  ;; that runs reclaims nothing and backs the backstop off, while one the
  ;; spacing holds back must leave the interval alone
  (test-case "a release the spacing holds back does not back the backstop off"
    (define (fired spacing)
      (settle!)
      (define before (collections))
      (parameterize ([native-memory-fraction 1/2]
                     [release-spacing spacing])
        (define held (for/list ([_ (in-range 60)]) (zeros 1024 1024)))
        (check-equal? (length held) 60))
      (- (collections) before))
    (install-device-queries! #:capacity (lambda (_dev) (* 128 mib))
                             #:allocated (lambda (_dev) (+ (cpu-bytes) (* 256 mib)))
                             #:release (lambda (_dev) #t))
    (define spaced (fired +inf.0))
    (define released (fired 0))
    (check-true (> spaced released)
                (format "~a backstop collections with the release held back, ~a without"
                        spaced released))
    (install-device-queries! #:capacity (lambda (_dev) #f)
                             #:allocated (lambda (_dev) #f))
    (settle!))

  ;; the stand-in's allocator holds the ledger plus `cache` bytes, which a
  ;; release may or may not give back
  (test-case "a release lowers the shadow by what it gave back, never raising it"
    (define cache 0)
    (define (run-backstop!)
      (parameterize ([native-memory-fraction 1/2]
                     [release-spacing 0])
        (define held (for/list ([_ (in-range 4)]) (zeros 1024 1024)))
        (check-equal? (length held) 4)))
    (define (install! empties?)
      (install-device-queries! #:capacity (lambda (_dev) (* 128 mib))
                               #:allocated (lambda (_dev) (+ (cpu-bytes) cache))
                               #:release (lambda (_dev)
                                           (when empties? (set! cache 0))
                                           #t)))
    (install! #t)
    (settle!)
    (set! cache (* 256 mib))
    (parameterize ([native-collect-margin idle-margin])
      (collect-at-trough!))
    (check-true (>= (cpu-unaccounted) (* 255 mib)) "the trough charged the cache")
    (run-backstop!)
    (check-equal? (cpu-unaccounted) 0 "the released cache is still charged")
    ;; growth the allocator shows only after the trough, as a graph's would
    (settle!)
    (set! cache 0)
    (parameterize ([native-collect-margin idle-margin])
      (collect-at-trough!))
    (install! #f)
    (set! cache (* 100 mib))
    (run-backstop!)
    (check-equal? (cpu-unaccounted) 0 "a release charged growth it observed")
    (install-device-queries! #:capacity (lambda (_dev) #f)
                             #:allocated (lambda (_dev) #f))
    (settle!))

  ;; the trough charged a cache that the next forward's tensors then reuse:
  ;; the allocator's reading stays put while the ledger grows
  (test-case "a sample lowers the shadow as tensors reuse the cache it charged"
    (define reading 0)
    (install-device-queries! #:capacity (lambda (_dev) #f)
                             #:allocated (lambda (_dev) reading))
    (settle!)
    (set! reading (+ (cpu-bytes) (* 256 mib)))
    (parameterize ([native-collect-margin idle-margin])
      (collect-at-trough!))
    (check-true (>= (cpu-unaccounted) (* 255 mib)) "the trough charged the cache")
    (parameterize ([native-memory-limit (* 1024 mib)])
      (define held (for/list ([_ (in-range 40)]) (zeros 1024 1024)))
      (check-equal? (length held) 40)
      (check-true (<= (cpu-unaccounted) (* 128 mib))
                  (format "~a MiB is charged both as tensors and as cache"
                          (quotient (cpu-unaccounted) mib))))
    (install-device-queries! #:capacity (lambda (_dev) #f)
                             #:allocated (lambda (_dev) #f))
    (settle!))

  (test-case "a sample that cannot read the allocator leaves the charge"
    (settle!)
    (parameterize ([allocator-reading (lambda (_dev) (+ (cpu-bytes) (* 256 mib)))]
                   [native-collect-margin idle-margin])
      (collect-at-trough!))
    (parameterize ([allocator-reading (lambda (_dev) #f)]
                   [native-memory-limit (* 1024 mib)])
      (define held (for/list ([_ (in-range 40)]) (zeros 1024 1024)))
      (check-equal? (length held) 40))
    (check-true (>= (cpu-unaccounted) (* 255 mib))
                "a failed reading lowered the charge as if it were zero")
    (settle!))

  ;; on this host emptying is a no-op, so the stand-in's cache is zeroed by
  ;; hand just before, as a real release would
  (test-case "emptying the cache by hand or for an OOM retry lowers the charge"
    (define cache 0)
    (define (charge-cache!)
      (settle!)
      (set! cache (* 256 mib))
      (parameterize ([native-collect-margin idle-margin])
        (collect-at-trough!))
      (check-true (>= (cpu-unaccounted) (* 255 mib)) "the trough charged the cache")
      (set! cache 0))
    (parameterize ([allocator-reading (lambda (_dev) (+ (cpu-bytes) cache))])
      (charge-cache!)
      (mps-empty-cache!)
      (check-equal? (cpu-unaccounted) 0 "mps-empty-cache! left the cache charged")
      (charge-cache!)
      (collect-and-drain!)
      (check-equal? (cpu-unaccounted) 0 "the OOM retry's drain left the cache charged"))
    (settle!))

  ;; the stand-in reports a cache shrunk to 200 MiB, then a tensor is
  ;; accounted before the ledger is read again, as another thread's could be
  (test-case "a lowering skips a reading the ledger moved under"
    (settle!)
    (parameterize ([allocator-reading (lambda (_dev) (+ (cpu-bytes) (* 256 mib)))]
                   [native-collect-margin idle-margin])
      (collect-at-trough!))
    (define late '())
    (parameterize ([allocator-reading
                    (lambda (_dev)
                      (define reading (+ (cpu-bytes) (* 200 mib)))
                      (set! late (cons (zeros 4096 4096) late))
                      reading)])
      (lower-shadows!))
    (check-equal? (length late) 1)
    (check-true (>= (cpu-unaccounted) (* 255 mib))
                (format "lowered to ~a MiB from a reading older than the ledger"
                        (quotient (cpu-unaccounted) mib)))
    (settle!)))
