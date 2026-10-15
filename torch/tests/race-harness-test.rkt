#lang racket/base

(module+ test
  (require rackunit
           (only-in "../foreign/raw/race-points.rkt" race-point)
           (only-in "private/race-harness.rkt"
                    await-roles! join join-all make-gate make-roster
                    release-pending? spawn stop-roles! until-stopped
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

  (test-case "a worker left parked never passes its race point"
    (for ([pool (in-list '(#f own))])
      (check-equal? (left-parked pool) '(#t #f #f) (format "~a" pool))))

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
    (check-true (andmap positive? completed) (format "~a" completed))
    (check-true (andmap positive? counts) (format "~a" counts)))

  (test-case "the start gate names the roles that never stepped at its deadline"
    (define r (make-roster))
    (define a (spawn (until-stopped r 'a void) #:pool 'own #:label 'a))
    (check-exn #rx"never completed a first step within 0.05 s: \\(b c\\)"
               (lambda () (await-roles! r '(a b c) #:seconds 0.05)))
    (check-true (positive? (join a #:seconds 30)))))
