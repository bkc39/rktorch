#lang racket/base

(module+ test
  (require (only-in ffi/unsafe register-finalizer)
           (only-in racket/place dynamic-place place-kill)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           "private/sandbox.rkt"
           (only-in "private/within.rkt" within))

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
                      (list (* workers per-worker) 1)))))

  (define-runtime-path place-finalizer-race "private/place-finalizer-race.rkt")

  (test-case "inside a place, parallel threads that register first share one finalizer thread"
    (within 300
            (lambda ()
              (for ([_ (in-range 3)])
                (define p (dynamic-place place-finalizer-race 'main))
                (define answer (sync/timeout 90 p))
                (place-kill p)
                (check-equal? answer (list (* workers per-worker) 1)))))))
