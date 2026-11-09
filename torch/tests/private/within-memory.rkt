#lang racket/base

(require (for-syntax (only-in syntax/parse/pre ...+ expr))
         (only-in racket/function const)
         (only-in rackunit fail test-case)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide memory-bounded-test
         within-memory)

(define (within-memory megabytes thunk #:seconds [seconds 60])
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
  (define outcome
    (sync/timeout seconds results (handle-evt worker (const 'shut-down))))
  (custodian-shutdown-all limited)
  (cond
    [(not outcome) (fail (format "no result within ~a s" seconds))]
    [(eq? outcome 'shut-down) (fail (format "shut down past its ~a MB memory limit" megabytes))]
    [else (outcome)]))

(define-syntax-parse-rule (memory-bounded-test name:expr body:expr ...+)
  (test-case name (within-memory 512 (lambda () body ...))))
