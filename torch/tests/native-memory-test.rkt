#lang racket/base

(module+ test
  (require rackunit
           (only-in racket/string string-split)
           (only-in "../foreign.rkt"
                    cpu-device native-memory-use ones reclaim-native-memory!
                    zeros)
           (only-in (submod "../foreign.rkt" unsafe) tensor-free!))

  (define (current-rss-bytes)
    (and (file-exists? "/proc/self/status")
         (call-with-input-file "/proc/self/status"
           (lambda (in)
             (for/first ([l (in-lines in)]
                         #:when (regexp-match? #rx"^VmRSS" l))
               (* 1024 (string->number (cadr (string-split l)))))))))

  (define (cpu-bytes)
    (cond [(assoc (cpu-device) (native-memory-use)) => cdr]
          [else 0]))

  (define (collect-until ready? #:tries [tries 50])
    (let loop ([i 0])
      (collect-garbage)
      (unless (or (ready?) (>= i tries))
        (sleep 0.01)
        (loop (add1 i)))))

  ;; unconditional rounds: earlier tests' finalizers may still be in flight
  (define (settled-baseline)
    (for ([_ (in-range 3)])
      (collect-garbage)
      (sleep 0.01))
    (cpu-bytes))

  (test-case "a live tensor is charged at its nbytes; GC releases it"
    (define base (settled-baseline))
    (define t (zeros 1024 1024)) ;; 4 MiB float32
    (check-true (>= (- (cpu-bytes) base) (* 4 1024 1024))
                "ledger charged the allocation")
    (set! t #f)
    (collect-until (lambda () (< (- (cpu-bytes) base) (* 1024 1024))))
    (check-true (< (- (cpu-bytes) base) (* 1024 1024))
                "ledger released after GC"))

  (test-case "the phantom charge is visible to the GC's own accounting"
    (define base (settled-baseline))
    (define before (current-memory-use))
    (define t (zeros 4096 4096)) ;; 64 MiB float32
    (check-true (> (- (current-memory-use) before) (* 32 1024 1024))
                "phantom pressure visible in current-memory-use")
    (tensor-free! t)
    (check-true (< (- (cpu-bytes) base) (* 1024 1024))
                "explicit free releases the charge immediately, no GC"))

  (test-case "explicit tensor-free! unaccounts synchronously"
    (define base (settled-baseline))
    (define t (ones 512 512)) ;; 1 MiB
    (check-true (>= (- (cpu-bytes) base) (* 1024 1024)))
    (tensor-free! t)
    ;; no collect-garbage on purpose — the free itself must unaccount
    (check-true (< (- (cpu-bytes) base) (* 512 1024))))

  (test-case "pressure triggers collection WITHOUT manual collects (#37)"
    (define base (settled-baseline))
    (define rss-before (current-rss-bytes))
    (define high-water
      (for/fold ([hw 0]) ([_ (in-range 200)])
        (void (zeros 1024 1024)) ;; 4 MiB, dropped
        (max hw (- (cpu-bytes) base))))
    (check-true (< high-water (* 600 1024 1024))
                (format "high-water ~a of ~a churned — pressure never fired"
                        high-water (* 800 1024 1024)))
    ;; RSS, not just the ledger: catches weak entries dropping while the C
    ;; free regressed (Linux only; /proc is absent on darwin)
    (when rss-before
      (check-true (< (- (current-rss-bytes) rss-before) (* 700 1024 1024))
                  "RSS grew by ~the whole churn — native buffers not freed")))

  (test-case "churn returns to baseline — the engagement regression guard"
    (define base (settled-baseline))
    (for ([_ (in-range 50)])
      (void (zeros 1024 1024))) ;; 50 x 4 MiB, all dropped
    (collect-until (lambda () (< (- (cpu-bytes) base) (* 8 1024 1024))))
    (check-true (< (- (cpu-bytes) base) (* 8 1024 1024))
                (format "ledger did not return toward baseline: ~a -> ~a"
                        base (cpu-bytes))))

  (test-case "reclaim-native-memory! drains dropped handles synchronously"
    (define base (settled-baseline))
    (for ([_ (in-range 25)])
      (void (zeros 1024 1024)))
    (reclaim-native-memory!)
    (check-true (< (- (cpu-bytes) base) (* 8 1024 1024))
                (format "reclaim left ledger bytes behind: ~a -> ~a"
                        base (cpu-bytes)))))
