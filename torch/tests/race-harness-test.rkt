#lang racket/base

(module+ test
  (require (only-in racket/format ~a)
           rackunit
           (only-in "../foreign/raw/race-points.rkt" race-point)
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/wedge.rkt" deadlines-armed status-and-stderr-of)
           (only-in "private/race-harness.rkt"
                    await-flag await-roles! join join-all make-gate make-roster pools
                    pools-stopped-by release-pending? spawn stop! stop-roles! stop-worker!
                    stops-for until-stopped wait-for-arrival while-inside with-parked
                    worker-thread))

  (define (left-parked pool)
    (define g (make-gate 'harness-parked))
    (define passed (box #f))
    (define a
      (with-parked [a (spawn (lambda ()
                               (race-point harness-parked #f)
                               (set-box! passed #t))
                             #:pool pool)
                    #:at g]
        (check-eq? (wait-for-arrival g) 'parked)
        a))
    (list (thread-dead? (worker-thread a)) (unbox passed) (release-pending? g)))

  (test-case "a worker that ends before its flag fails the wait at once with its error"
    (for ([pool (in-list pools)])
      (define a (spawn (lambda () (error 'worker "died before its flag")) #:pool pool))
      (check-exn #rx"died before its flag" (lambda () (await-flag a (box #f) #:seconds 30))
                 (~a pool))))

  (define (flag-set-after-first-read)
    (define go (make-semaphore 0))
    (define flag (box #f))
    (define a (spawn (lambda () (semaphore-wait go) (set-box! flag #t))))
    (define first-read? #t)
    (define (read-then-let-it-finish _b seen)
      (when first-read?
        (set! first-read? #f)
        (semaphore-post go)
        (thread-wait (worker-thread a)))
      seen)
    (await-flag a (impersonate-box flag read-then-let-it-finish (lambda (_b v) v))))

  (test-case "a worker that sets its flag and ends between two polls has set its flag"
    (check-true (flag-set-after-first-read)))

  (test-case "a worker that misses its flag's deadline is stopped before the wait fails"
    (for ([pool (in-list pools)])
      (define a (spawn (lambda () (sync never-evt)) #:pool pool))
      (check-exn #rx"within 0.2 s" (lambda () (await-flag a (box #f) #:seconds 0.2)) (~a pool))
      (check-true (thread-dead? (worker-thread a)) (~a pool))))

  (test-case "a thread of its own pool is stopped by kill, never by break (#443)"
    (define a (spawn (lambda () (sync never-evt)) #:pool 'own))
    (check-exn exn:fail:contract? (lambda () (stop-worker! a 'break)))
    (check-false (thread-dead? (worker-thread a)) "a refused break reached the thread")
    (stop-worker! a 'kill)
    (check-true (thread-dead? (worker-thread a))))

  (test-case "the stop policy and the flag wait check their arguments at the boundary"
    (define blamed? (blames-caller? "race-harness-test.rkt"))
    (check-exn blamed? (lambda () (stops-for 'shared)))
    (check-exn blamed? (lambda () (pools-stopped-by 'interrupt)))
    (define a (spawn (lambda () (sync never-evt))))
    (check-exn blamed? (lambda () (await-flag a #t)))
    (check-exn blamed? (lambda () (await-flag a (box #f) #:seconds 0)))
    (check-false (thread-dead? (worker-thread a)) "a refused wait stopped the worker")
    (stop! a)
    (check-equal? (pools-stopped-by 'break) '(#f)))

  (test-case "a green thread is stopped by break"
    (define running (box #f))
    (define a (spawn (lambda () (set-box! running #t) (sync never-evt))))
    (await-flag a running)
    (stop-worker! a 'break)
    (check-exn exn:break? (lambda () (join a #:seconds 10))))

  (test-case "a worker left parked never passes its race point"
    (for ([pool (in-list '(#f own))])
      (check-equal? (left-parked pool) '(#t #f #f) (~a pool))))

  (test-case "stopping workers that have already ended arms no process deadline"
    (for ([pool (in-list pools)])
      (define ended (spawn void #:pool pool))
      (join ended)
      (check-equal? (deadlines-armed (lambda () (stop! ended))) 0 (~a pool))
      (check-equal? (deadlines-armed (lambda () (while-inside (lambda (k) (k)) void #:pool pool)))
                    0 (~a pool))
      (define running (spawn (lambda () (sync never-evt)) #:pool pool))
      (check-equal? (deadlines-armed (lambda () (stop! running))) 1 (~a pool))))

  (define (slow-first-step done)
    (define first? #t)
    (lambda ()
      (cond
        [first? (set! first? #f) (sleep 0.2)]
        [else (void)])
      (set-box! done (add1 (unbox done)))))

  (test-case "the start gate opens only after every role has completed a step"
    (define r (make-roster))
    (define labels '(a b c))
    (define done (map (lambda (_) (box 0)) labels))
    (define workers
      (for/list ([label (in-list labels)]
                 [d (in-list done)])
        (spawn (until-stopped r label (slow-first-step d))
               #:pool 'own
               #:label label)))
    (await-roles! r labels #:seconds 30)
    (define completed (map unbox done))
    (stop-roles! r)
    (define counts (join-all workers #:seconds 30))
    (check-true (andmap positive? completed) (~a completed))
    (check-true (andmap positive? counts) (~a counts)))

  (test-case "the start gate names the roles that never stepped at its deadline"
    (define r (make-roster))
    (define a (spawn (until-stopped r 'a void) #:pool 'own #:label 'a))
    (check-exn #rx"never completed a first step within 0.05 s: \\(b c\\)"
               (lambda () (await-roles! r '(a b c) #:seconds 0.05)))
    (check-true (positive? (join a #:seconds 30))))

  (define (wedged pool)
    (spawn (lambda () (let idle () (sleep 0.01) (idle))) #:pool pool))

  (test-case "a wedged worker set fails within one deadline, not one per worker"
    (for ([pool (in-list pools)])
      (define workers (for/list ([_ (in-range 4)]) (wedged pool)))
      (define start (current-inexact-monotonic-milliseconds))
      (check-exn #rx"did not finish within 0.5 s"
                 (lambda () (join-all workers #:seconds 0.5)) (~a pool))
      (define ms (- (current-inexact-monotonic-milliseconds) start))
      (check-true (< ms 1500) (format "join-all took ~a ms (pool ~a)" ms pool))
      (check-true (andmap (lambda (w) (thread-dead? (worker-thread w))) workers) (~a pool))))

  (test-case "workers that each finish inside the bound but together past it fail the join"
    (for ([pool (in-list pools)])
      (define workers
        (for/list ([k (in-range 4)])
          (spawn (lambda () (sleep (* 0.3 (add1 k))) k) #:pool pool)))
      (define start (current-inexact-monotonic-milliseconds))
      (check-exn #rx"did not finish within 0.5 s"
                 (lambda () (join-all workers #:seconds 0.5)) (~a pool))
      (define ms (- (current-inexact-monotonic-milliseconds) start))
      (check-true (< ms 1000) (format "join-all took ~a ms (pool ~a)" ms pool))))

  (test-case "a worker that raises stops the rest of the set"
    (for ([pool (in-list pools)])
      (define rest (for/list ([_ (in-range 3)]) (wedged pool)))
      (define raiser (spawn (lambda () (error 'worker "boom")) #:pool pool))
      (check-exn #rx"boom" (lambda () (join-all (cons raiser rest) #:seconds 30)) (~a pool))
      (check-true (andmap (lambda (w) (thread-dead? (worker-thread w))) rest) (~a pool))))

  (test-case "a worker that raises fails the join at once, wherever it sits in the set"
    (for ([pool (in-list pools)])
      (define posted (make-semaphore 0))
      (define waiter (spawn (lambda () (semaphore-wait posted)) #:pool pool))
      (define raiser (spawn (lambda () (error 'worker "boom") (semaphore-post posted)) #:pool pool))
      (define start (current-inexact-monotonic-milliseconds))
      (check-exn #rx"boom" (lambda () (join-all (list waiter raiser) #:seconds 2)) (~a pool))
      (define ms (- (current-inexact-monotonic-milliseconds) start))
      (check-true (< ms 1500) (format "join-all took ~a ms (pool ~a)" ms pool))
      (check-true (thread-dead? (worker-thread waiter)) (~a pool))))

  (define harness-module
    (path->string (collection-file-path "race-harness.rkt" "torch" "tests" "private")))

  (define (stopped-in-child body)
    (status-and-stderr-of
     `(begin
        (module run racket/base
          (require (only-in ffi/unsafe _fun _uint get-ffi-obj)
                   (only-in (file ,harness-module) join join-all spawn stop!))
          (define c-sleep (get-ffi-obj "sleep" #f (_fun #:blocking? #t _uint -> _uint)))
          (define (sleeper)
            (define inside (make-semaphore 0))
            (begin0 (spawn (lambda () (semaphore-post inside) (c-sleep 600)) #:pool 'own)
                    (sync/timeout 30 inside)
                    (sleep 0.5)))
          ,body)
        (require 'run))
     #:within 60
     #:rerun-unless #rx"process deadline: "))

  (test-case "stopping a set whose worker sits in a foreign call ends the process, named"
    (define-values (status stderr)
      (stopped-in-child '(join-all (list (spawn (lambda () (error 'worker "boom"))) (sleeper))
                                   #:seconds 30 #:stop-seconds 2)))
    (check-equal? status (+ 128 14) stderr)
    (check-regexp-match #rx"process deadline: stopping a worker set ran past 2 s" stderr))

  (test-case "a set that times out with a worker in a foreign call ends the process, named"
    (define-values (status stderr)
      (stopped-in-child '(join-all (list (sleeper)) #:seconds 0.5 #:stop-seconds 2)))
    (check-equal? status (+ 128 14) stderr)
    (check-regexp-match #rx"process deadline: stopping a worker set ran past 2 s" stderr))

  (test-case "stopping or joining a worker in a foreign call ends the process, named"
    (for ([body (in-list '((stop! (sleeper) #:stop-seconds 2)
                           (join (sleeper) #:seconds 0.5 #:stop-seconds 2)))])
      (define-values (status stderr) (stopped-in-child body))
      (check-equal? status (+ 128 14) (format "~s: ~a" body stderr))
      (check-regexp-match #rx"process deadline: stopping a worker ran past 2 s" stderr))))
