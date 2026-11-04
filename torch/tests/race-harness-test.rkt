#lang racket/base

(module+ test
  (require (only-in racket/format ~a)
           rackunit
           (only-in "../foreign/raw/race-points.rkt" race-point)
           (only-in "private/race-harness.rkt"
                    await-flag await-roles! join join-all make-gate make-roster pools
                    release-pending? spawn stop-roles! stop-worker! until-stopped
                    wait-for-arrival with-parked worker-thread))

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

  (test-case "a green thread is stopped by break"
    (define running (box #f))
    (define a (spawn (lambda () (set-box! running #t) (sync never-evt))))
    (await-flag a running)
    (stop-worker! a 'break)
    (check-exn exn:break? (lambda () (join a #:seconds 10))))

  (test-case "a worker left parked never passes its race point"
    (for ([pool (in-list '(#f own))])
      (check-equal? (left-parked pool) '(#t #f #f) (~a pool))))

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
    (check-true (positive? (join a #:seconds 30)))))
