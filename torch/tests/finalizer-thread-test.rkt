#lang racket/base

(module+ test
  (require (only-in ffi/unsafe register-finalizer)
           rackunit
           "private/sandbox.rkt")

  (define workers 8)
  (define per-worker 2000)

  (test-case "parallel threads whose first registration is the drain's canary share one finalizer thread"
    (for ([_ (in-range 3)])
      (with-fresh-eval (ev #:time-limit 60)
        (check-false (eq? (ev '(dynamic-require 'ffi/unsafe 'register-finalizer))
                          register-finalizer)
                     "the sandbox has its own finalizer thread to start")
        (ev '(require torch/tests/private/finalizer-race))
        (check-equal? (ev `(run-race ,workers ,per-worker))
                      (list (* workers per-worker) 1))))))
