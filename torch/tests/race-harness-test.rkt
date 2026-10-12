#lang racket/base

(module+ test
  (require rackunit
           (only-in "../foreign/raw/race-points.rkt" race-point)
           (only-in "private/race-harness.rkt"
                    make-gate release-pending? spawn wait-for-arrival
                    with-parked worker-thread))

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
      (check-equal? (left-parked pool) '(#t #f #f) (format "~a" pool)))))
