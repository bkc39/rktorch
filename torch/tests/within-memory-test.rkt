#lang racket/base

(module+ test
  (require (only-in racket/function const)
           rackunit
           (only-in "private/within-memory.rkt" within-memory))

  (test-case "a timed-out within-memory stops its body"
    (define count 0)
    (define (spin)
      (set! count (add1 count))
      (spin))
    (check-eq? (with-handlers ([exn:test:check? (const 'timed-out)])
                 (within-memory 64 spin #:seconds 0.2))
               'timed-out)
    (define stopped count)
    (sleep 0.2)
    (check-equal? count stopped))

  (test-case "within-memory returns the body's values"
    (check-equal? (call-with-values (lambda () (within-memory 64 (lambda () (values 1 2))))
                                    list)
                  '(1 2))))
