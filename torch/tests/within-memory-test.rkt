#lang racket/base

(module+ test
  (require (only-in racket/function const)
           rackunit
           (only-in "private/wedge.rkt" status-and-stderr-of)
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
                  '(1 2)))

  (define within-memory-module
    (path->string (collection-file-path "within-memory.rkt" "torch" "tests" "private")))

  (test-case "an allocating worker the body leaves running is stopped before within-memory returns"
    (define-values (status stderr)
      (status-and-stderr-of
       `(begin
          (module run racket/base
            (require (only-in (file ,within-memory-module) within-memory))
            (define worker (box #f))
            (define (allocate)
              (make-vector 1000)
              (allocate))
            (within-memory 64 (lambda ()
                                (set-box! worker (thread allocate #:pool 'own))
                                (sleep 0.1)))
            (eprintf "worker dead: ~a" (thread-dead? (unbox worker))))
          (require 'run))
       #:within 30))
    (check-equal? status 0 stderr)
    (check-equal? stderr "worker dead: #t")))
