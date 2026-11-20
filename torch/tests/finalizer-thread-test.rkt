#lang racket/base

(module+ test
  (require (only-in racket/list make-list)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in "private/child.rkt" run-child))

  (define workers 8)
  (define per-worker 2000)

  (define-runtime-path place-finalizer-race "private/place-finalizer-race.rkt")
  (define-runtime-path place-stall "private/place-stall.rkt")
  (define-runtime-path sandbox "private/sandbox.rkt")
  (define-runtime-path wedge "private/wedge.rkt")

  (define (answer-in-child requires body #:seconds seconds)
    (run-child
     `(begin
        (require (only-in (file ,(path->string wedge)) call-with-process-deadline) ,@requires)
        (write (call-with-process-deadline ,(+ seconds 5) (lambda () ,@body))))
     #:within (+ seconds 6)))

  (test-case "parallel threads whose first registration is the drain's canary share one finalizer thread"
    (define-values (status answers stderr)
      (answer-in-child
       `((only-in ffi/unsafe register-finalizer)
         (only-in (file ,(path->string sandbox)) with-fresh-eval))
       `((for/list ([_ (in-range 3)])
           (with-fresh-eval (ev #:time-limit 25)
             (define thread-of (if (eq? (ev '(dynamic-require 'ffi/unsafe 'register-finalizer))
                                        register-finalizer)
                                   'the-parent
                                   'its-own))
             (ev '(require torch/tests/private/finalizer-race))
             (list thread-of (ev '(run-race ,workers ,per-worker))))))
       #:seconds 25))
    (check-equal? status 0 stderr)
    (check-equal? answers (make-list 3 (list 'its-own (list (* workers per-worker) 1)))
                  "each sandbox has its own finalizer thread to start, and one runs the race"))

  (define (place-answers-in-child place entry runs #:seconds seconds)
    (answer-in-child
     '((only-in racket/place dynamic-place place-dead-evt place-kill))
     `((define deadline (+ (current-inexact-monotonic-milliseconds) ,(* 1000 seconds)))
       (define (answer)
         (define p (dynamic-place (string->path ,(path->string place)) ',entry))
         (define left (max 0 (/ (- deadline (current-inexact-monotonic-milliseconds)) 1000.0)))
         (begin0
           (sync/timeout left p (handle-evt (place-dead-evt p) (lambda (_) (sync/timeout 0 p))))
           (place-kill p)))
       (for/list ([_ (in-range ,runs)]) (answer)))
     #:seconds seconds))

  (test-case "inside a place, parallel threads that register first share one finalizer thread"
    (define-values (status answers stderr)
      (place-answers-in-child place-finalizer-race 'main 3 #:seconds 25))
    (check-equal? status 0 stderr)
    (check-equal? answers (make-list 3 (list (* workers per-worker) 1)) stderr))

  (test-case "a place that ends or stalls without answering fails its run within the bound"
    (for ([entry (in-list '(quit main))]
          [seconds (in-list '(20 2))])
      (define start (current-inexact-monotonic-milliseconds))
      (define-values (status answers stderr)
        (place-answers-in-child place-stall entry 1 #:seconds seconds))
      (check-equal? status 0 stderr)
      (check-equal? answers '(#f) (format "~a" entry))
      (check-true (< (- (current-inexact-monotonic-milliseconds) start) 10000)
                  (format "~a" entry)))))
