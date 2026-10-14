#lang racket/base

(module+ test
  (require (only-in racket/match match match-define)
           rackunit
           (only-in "../foreign.rkt"
                    backward! finalizer-diagnostics native-memory-limit ones
                    sum)
           (only-in "../nn.rkt" Linear)
           (only-in "private/race-harness.rkt"
                    check-ledger churn-once! env-number join-all
                    ledger-snapshot make-watch seeded-generator settle! spawn
                    watch-hook with-race-hook))

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

  (define (count-in! counter)
    (define n (unbox counter))
    (unless (box-cas! counter n (add1 n))
      (count-in! counter)))

  (define (await-count! counter n)
    (define deadline
      (+ (current-inexact-milliseconds) (* 1000 start-deadline-seconds)))
    (let loop ()
      (cond
        [(>= (unbox counter) n) (void)]
        [(> (current-inexact-milliseconds) deadline) (void)]
        [else (sleep 0.001) (loop)])))

  (define ((until-stopped started stop? step!))
    (count-in! started)
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

  (define (role-step label)
    (match label
      ['trainer (trainer-step)]
      ['adversary (adversary-step)]
      [k (worker-step k)]))

  (define (phase width)
    (settle!)
    (define base (ledger-snapshot))
    (define entries (ledger-entries))
    (define before (diagnostic 'pressure-collections))
    (define w (make-watch))
    (define stop? (box #f))
    (define started (box 0))
    (define labels (append (build-list width values) '(trainer adversary)))
    (define counts
      (with-race-hook (watch-hook w)
        (define workers
          (parameterize ([native-memory-limit (quotient mib 4)])
            (for/list ([label (in-list labels)])
              (spawn (until-stopped started stop? (role-step label))
                     #:pool 'own
                     #:label label))))
        (await-count! started (length labels))
        (sleep seconds-per-phase)
        (set-box! stop? #t)
        (begin0 (join-all workers #:seconds 300)
                (settle!))))
    (match-define (list worker-steps ... trainer-steps _adversary-steps) counts) ;; noqa
    (printf "stress: ~a workers, ~a steps, ~a trainer steps, ~a collections\n"
            width
            (apply + worker-steps)
            trainer-steps
            (- (diagnostic 'pressure-collections) before))
    (check-ledger base w)
    (check-equal? (ledger-entries) entries "ledger entries left behind")
    (for ([label (in-list labels)]
          [n (in-list counts)])
      (check-true (positive? n) (format "role ~a never stepped" label))))

  (test-case "parallel threads churning tensors keep the ledger invariants"
    (define rss-before (rss-bytes))
    (for ([width (in-list widths)])
      (with-check-info (['workers width])
        (phase width)))
    (define rss-after (rss-bytes))
    (cond
      [rss-before
       (printf "stress: RSS ~a MiB -> ~a MiB\n"
               (quotient rss-before mib) (quotient rss-after mib))
       (check-true (< (- rss-after rss-before) (* 512 mib))
                   "RSS grew past the churn's working set")]
      [else (void)])))
