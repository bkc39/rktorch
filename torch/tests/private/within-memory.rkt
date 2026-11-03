#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in racket/function const)
         (only-in rackunit fail)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "within.rkt" bounded-test))

(provide memory-bounded-test
         within-memory)

(define (within-memory megabytes thunk)
  (define limited (make-custodian))
  (custodian-limit-memory limited (* megabytes 1024 1024) limited)
  (define results (make-channel))
  (define worker
    (parameterize ([current-custodian limited])
      (thread (lambda ()
                (channel-put results
                             (with-handlers ([(const #t) (lambda (v) (lambda () (raise v)))])
                               (call-with-values thunk
                                                 (lambda vs (lambda () (apply values vs))))))))))
  (define outcome (sync results (handle-evt worker (const #f))))
  (custodian-shutdown-all limited)
  (cond
    [outcome (outcome)]
    [else (fail (format "shut down past its ~a MB memory limit" megabytes))]))

(define-syntax-parse-rule (memory-bounded-test name:expr body:expr ...+)
  (bounded-test name (within-memory 512 (lambda () body ...))))
