#lang racket/base

(module+ test
  (require (only-in racket/match match match-define)
           rackunit
           (only-in "../foreign.rkt"
                    backward! finalizer-diagnostics mul! native-memory-limit ones
                    sum)
           (only-in "../nn.rkt" Linear)
           (only-in "private/race-harness.rkt"
                    await-roles! check-ledger churn-once! env-number join-all
                    ledger-snapshot make-roster make-watch seeded-generator
                    settle! spawn stop-roles! until-stopped watch-hook
                    with-race-hook))

  (define mib (* 1024 1024))

  (define seconds-per-phase (env-number "RKTORCH_STRESS_SECONDS" 0.5))

  (define widths '(2 4 8 16))

  (define start-deadline-seconds 60)

  (define (rss-bytes)
    (define status "/proc/self/status")
    (and (file-exists? status)
         (call-with-input-file status
           (lambda (in)
             (for/first ([line (in-lines in)]
                         #:when (regexp-match? #rx"^VmRSS:" line))
               (match-define (list _ kb) (regexp-match #rx"([0-9]+) kB" line))
               (* 1024 (string->number kb)))))))

  (define (diagnostic key)
    (cdr (assq key (finalizer-diagnostics))))

  (define (ledger-entries)
    (diagnostic 'ledger-entries))

  (define (worker-step k)
    (define rng (seeded-generator 40 (list 'stress k)))
    (lambda () (churn-once! rng)))

  (define (trainer-step)
    (define net (Linear 32 32))
    (define x (ones 8 32))
    (lambda () (backward! (sum (net x)))))

  (define (adversary-step)
    (define n 0)
    (lambda ()
      (set! n (add1 n))
      (collect-garbage (if (zero? (modulo n 8)) 'major 'minor))
      (sleep 0.002)))

  (define (role-step label)
    (match label
      ['trainer (trainer-step)]
      ['adversary (adversary-step)]
      [k (worker-step k)]))

  (define (sleep-through-phase)
    (sleep seconds-per-phase))

  ;; Main in plain foreign calls, yielding only every 5 ms: the workers'
  ;; ops must not wait for it.
  (define (foreign-calls-through-phase)
    (define t (ones 64 64))
    (define end (+ (current-inexact-milliseconds) (* 1000 seconds-per-phase)))
    (let loop ([next-yield (+ (current-inexact-milliseconds) 5)])
      (define now (current-inexact-milliseconds))
      (cond
        [(>= now end) (void)]
        [(>= now next-yield) (sleep 0) (loop (+ now 5))]
        [else (mul! t 1.0) (loop next-yield)])))

  (define (phase labels meanwhile)
    (settle!)
    (define base (ledger-snapshot))
    (define entries (ledger-entries))
    (define before (diagnostic 'pressure-collections))
    (define w (make-watch))
    (define r (make-roster))
    (define counts
      (with-race-hook (watch-hook w)
        (define workers
          (parameterize ([native-memory-limit (quotient mib 4)])
            (for/list ([label (in-list labels)])
              (spawn (until-stopped r label (role-step label))
                     #:pool 'own
                     #:label label))))
        (await-roles! r labels #:seconds start-deadline-seconds)
        (meanwhile)
        (stop-roles! r)
        (begin0 (join-all workers #:seconds 300)
                (settle!))))
    (define worker-steps
      (for/sum ([label (in-list labels)] [n (in-list counts)] #:when (exact-integer? label))
        n))
    (printf "stress: ~a, ~a worker steps, ~a collections\n"
            labels worker-steps (- (diagnostic 'pressure-collections) before))
    (check-ledger base w)
    (check-equal? (ledger-entries) entries "ledger entries left behind")
    (for ([label (in-list labels)]
          [n (in-list counts)])
      (check-true (positive? n) (format "role ~a never stepped" label))))

  (test-case "parallel threads churning tensors keep the ledger invariants"
    (define rss-before (rss-bytes))
    (for ([width (in-list widths)])
      (with-check-info (['workers width])
        (phase (append (build-list width values) '(trainer adversary))
               sleep-through-phase)))
    (with-check-info (['workers 8] ['main 'foreign-calls])
      (phase (build-list 8 values) foreign-calls-through-phase))
    (define rss-after (rss-bytes))
    (cond
      [rss-before
       (printf "stress: RSS ~a MiB -> ~a MiB\n"
               (quotient rss-before mib) (quotient rss-after mib))
       (check-true (< (- rss-after rss-before) (* 512 mib))
                   "RSS grew past the churn's working set")]
      [else (void)])))
