#lang racket/base

;; Leg 1.5 (#38): typed OOM errors + the collect-and-retry mechanism.
;;
;; The end-to-end cases lean on the one portably provokable exhaustion:
;; a CPU request beyond user address space (the allocator rejects it
;; upfront — no overcommit-then-OOM-killer hazard; the same shape the
;; C-side gtest CpuOomClassifiesAsOomKind pins). The retry mechanism is
;; tested at the combinator level with injected probes, matching
;; finalizer-guard-test.rkt's style: provoking a real
;; fails-once-then-succeeds exhaustion would need a full memory squeeze,
;; which belongs to scripts/oom-repro.rkt, not the unit suite.

(module+ test
  (require rackunit
           "../main.rkt"
           (only-in "../foreign/raw/memory.rkt" oom-retry))

  ;; 2^40 x 2^10 = 2^50 floats = 4 PiB: beyond any host's address space.
  (define (absurd-alloc!)
    (zeros 1099511627776 1024))

  (test-case "CPU exhaustion raises the typed exn"
    (define e
      (with-handlers ([exn:fail:rktorch:oom? values])
        (absurd-alloc!)
        (fail "absurd allocation unexpectedly succeeded")))
    (check-pred exn:fail:rktorch:oom? e)
    ;; the typed exn IS an exn:fail — existing catch-all handlers keep
    ;; working
    (check-pred exn:fail? e)
    (check-regexp-match #rx"zeros" (exn-message e)))

  (test-case "non-OOM failures stay plain exn:fail"
    (define e
      (with-handlers ([exn:fail? values])
        (reshape (zeros 2 2) 3 5)
        (fail "bad reshape unexpectedly succeeded")))
    (check-pred exn:fail? e)
    (check-false (exn:fail:rktorch:oom? e))
    ;; ...including AFTER an OOM: the kind resets with each recording
    (with-handlers ([exn:fail:rktorch:oom? void]) (absurd-alloc!))
    (define e2
      (with-handlers ([exn:fail? values])
        (reshape (zeros 2 2) 3 5)))
    (check-false (exn:fail:rktorch:oom? e2)))

  ;; --- the retry combinator, mechanism level (injected probes) ----------

  (test-case "oom-retry: NULL + oom kind collects and retries exactly once"
    (define calls 0)
    (define collects 0)
    (define (fails-once . _args)
      (set! calls (add1 calls))
      (if (= calls 1) #f 'handle))
    (define wrapped
      ((oom-retry #:oom? (lambda () #t)
                  #:collect! (lambda () (set! collects (add1 collects))))
       fails-once))
    (check-equal? (wrapped 'arg) 'handle)
    (check-equal? calls 2)
    (check-equal? collects 1))

  (test-case "oom-retry: a second failure surfaces as NULL (caller raises)"
    (define collects 0)
    (define wrapped
      ((oom-retry #:oom? (lambda () #t)
                  #:collect! (lambda () (set! collects (add1 collects))))
       (lambda _args #f)))
    (check-false (wrapped 'arg))
    ;; exactly one collect — no retry storm
    (check-equal? collects 1))

  (test-case "oom-retry: non-OOM failures never collect or retry"
    (define calls 0)
    (define collects 0)
    (define wrapped
      ((oom-retry #:oom? (lambda () #f)
                  #:collect! (lambda () (set! collects (add1 collects))))
       (lambda _args (set! calls (add1 calls)) #f)))
    (check-false (wrapped 'arg))
    (check-equal? calls 1)
    (check-equal? collects 0))

  (test-case "oom-retry: success path touches neither probe"
    (define collects 0)
    (define wrapped
      ((oom-retry #:oom? (lambda () (error "kind probe on success path"))
                  #:collect! (lambda () (set! collects (add1 collects))))
       (lambda _args 'handle)))
    (check-equal? (wrapped 'arg) 'handle)
    (check-equal? collects 0))

  ;; --- seeded parity: the RNG family is excluded from the retry ---------

  (test-case "randn draws an identical stream across an interleaved OOM"
    (manual-seed! 42)
    (define a (tensor->list (randn 4)))
    (manual-seed! 42)
    ;; an OOM (with its internal collect + retry) between draws must not
    ;; advance the generator
    (with-handlers ([exn:fail:rktorch:oom? void]) (absurd-alloc!))
    (define b (tensor->list (randn 4)))
    (check-equal? b a)))
