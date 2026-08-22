#lang racket/base

(module+ test
  (require rackunit
           (only-in "../generated.rkt" dropout)
           "../main.rkt"
           (only-in "../foreign/raw/memory.rkt" oom-retry))

  ;; 2^60 floats = 4 EiB: beyond any 64-bit user address space (so the
  ;; failure is deterministic — no overcommit/OOM-killer hazard) yet below
  ;; INT64_MAX, so ATen's numel arithmetic can't overflow into a different
  ;; error shape.
  (define (absurd-alloc!)
    (zeros 1152921504606846976))

  (test-case "CPU exhaustion raises the typed exn (still an exn:fail)"
    (define e
      (with-handlers ([exn:fail:rktorch:oom? values])
        (absurd-alloc!)
        (fail "absurd allocation unexpectedly succeeded")))
    (check-pred exn:fail:rktorch:oom? e)
    (check-pred exn:fail? e)
    (check-regexp-match #rx"zeros" (exn-message e)))

  (test-case "randn OOM surfaces typed through the no-retry RNG path"
    (check-pred exn:fail:rktorch:oom?
                (with-handlers ([exn:fail? values])
                  (randn 1152921504606846976)
                  (fail "absurd randn unexpectedly succeeded"))))

  (test-case "non-OOM failures stay plain exn:fail, before and after an OOM"
    (define e
      (with-handlers ([exn:fail? values])
        (reshape (zeros 2 2) 3 5)
        (fail "bad reshape unexpectedly succeeded")))
    (check-pred exn:fail? e)
    (check-false (exn:fail:rktorch:oom? e))
    (with-handlers ([exn:fail:rktorch:oom? void]) (absurd-alloc!))
    (define e2
      (with-handlers ([exn:fail? values])
        (reshape (zeros 2 2) 3 5)
        (fail "bad reshape unexpectedly succeeded")))
    (check-pred exn:fail? e2)
    (check-false (exn:fail:rktorch:oom? e2)))

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

  (test-case "randn draws an identical stream across an interleaved OOM"
    (manual-seed! 42)
    (define a (tensor->list (randn 4)))
    (manual-seed! 42)
    (with-handlers ([exn:fail:rktorch:oom? void]) (absurd-alloc!))
    (define b (tensor->list (randn 4)))
    (check-equal? b a))

  (test-case "dropout (generated #:rng arm) draws identically across an OOM"
    (define x (ones 64))
    (manual-seed! 7)
    (define a (tensor->list (dropout x 0.5 #t)))
    (manual-seed! 7)
    (with-handlers ([exn:fail:rktorch:oom? void]) (absurd-alloc!))
    (define b (tensor->list (dropout x 0.5 #t)))
    (check-equal? b a)))
