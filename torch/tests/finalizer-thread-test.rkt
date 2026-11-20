#lang racket/base

(module+ test
  (require (only-in ffi/unsafe register-finalizer)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in "private/child.rkt" run-child)
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
                      (list (* workers per-worker) 1)))))

  (define-runtime-path place-finalizer-race "private/place-finalizer-race.rkt")
  (define-runtime-path place-stall "private/place-stall.rkt")
  (define-runtime-path wedge "private/wedge.rkt")

  (define (place-answer-in-child place #:seconds seconds)
    (run-child
     `(begin
        (require (only-in racket/place dynamic-place place-kill)
                 (only-in (file ,(path->string wedge)) call-with-process-deadline))
        (call-with-process-deadline
         ,seconds
         (lambda ()
           (define p (dynamic-place (string->path ,(path->string place)) 'main))
           (write (sync/timeout ,(* 2 seconds) p))
           (place-kill p))))))

  (test-case "inside a place, parallel threads that register first share one finalizer thread"
    (for ([_ (in-range 3)])
      (define-values (status answer stderr) (place-answer-in-child place-finalizer-race
                                                                   #:seconds 100))
      (check-equal? status 0 stderr)
      (check-equal? answer (list (* workers per-worker) 1) stderr)))

  (test-case "a place that never answers fails the test, not the process running it"
    (define-values (status _answer stderr) (place-answer-in-child place-stall #:seconds 2))
    (check-equal? status (+ 128 14))
    (check-regexp-match #rx"process deadline: .* ran past 2 s" stderr)))
