#lang racket/base

(module+ test
  (require rackunit
           (only-in "../foreign.rkt"
                    backward! finalizer-diagnostics native-memory-limit ones
                    sum)
           (only-in "../nn.rkt" Linear)
           (only-in "private/race-harness.rkt"
                    check-ledger churn-once! hooks join-all ledger-snapshot
                    make-watch seeded-generator settle! spawn watch-hook
                    with-race-hook))

  (define mib (* 1024 1024))

  (define seconds-per-phase
    (let ([text (getenv "RKTORCH_STRESS_SECONDS")])
      (or (and text (string->number text)) 0.5)))

  (define widths '(2 4 8 16))

  (define (ledger-entries)
    (cdr (assq 'ledger-entries (finalizer-diagnostics))))

  (define (rss-bytes)
    (define status "/proc/self/status")
    (and (file-exists? status)
         (call-with-input-file status
           (lambda (in)
             (for/first ([line (in-lines in)]
                         #:when (regexp-match? #rx"^VmRSS:" line))
               (* 1024 (string->number
                        (cadr (regexp-match #rx"([0-9]+) kB" line)))))))))

  (define (diagnostic key)
    (cdr (assq key (finalizer-diagnostics))))

  (define ((until-stopped stop? step!))
    (let loop ([n 0])
      (cond
        [(unbox stop?) n]
        [else (step!) (loop (add1 n))])))

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

  (define (phase width)
    (settle!)
    (define base (ledger-snapshot))
    (define entries (ledger-entries))
    (define before (diagnostic 'pressure-collections))
    (define w (make-watch))
    (define stop? (box #f))
    (define counts
      (with-race-hook (hooks (watch-hook w))
        (define workers
          (parameterize ([native-memory-limit (quotient mib 4)])
            (append
             (for/list ([k (in-range width)])
               (spawn (until-stopped stop? (worker-step k)) #:pool 'own #:label k))
             (list (spawn (until-stopped stop? (trainer-step)) #:pool 'own
                          #:label 'trainer)
                   (spawn (until-stopped stop? (adversary-step)) #:pool 'own
                          #:label 'adversary)))))
        (sleep seconds-per-phase)
        (set-box! stop? #t)
        (begin0 (join-all workers #:seconds 300)
                (settle!))))
    (printf "stress: ~a workers, ~a steps, ~a trainer steps, ~a collections\n"
            width
            (apply + (reverse (cddr (reverse counts))))
            (list-ref counts width)
            (- (diagnostic 'pressure-collections) before))
    (check-ledger base w)
    (check-equal? (ledger-entries) entries "ledger entries left behind")
    (check-true (positive? (list-ref counts width)) "the trainer never stepped"))

  (test-case "parallel threads churning tensors keep the ledger invariants"
    (define rss-before (rss-bytes))
    (for ([width (in-list widths)])
      (with-check-info (['workers width])
        (phase width)))
    (define rss-after (rss-bytes))
    (when rss-before
      (printf "stress: RSS ~a MiB -> ~a MiB\n"
              (quotient rss-before mib) (quotient rss-after mib))
      (check-true (< (- rss-after rss-before) (* 512 mib))
                  "RSS grew past the churn's working set"))))
