#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in racket/function const)
         (only-in rackunit fail test-case)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide bounded-test
         within)

(define (within seconds thunk)
  (define outcome #f)
  (define t
    (thread (lambda ()
              (set! outcome
                    (with-handlers ([(const #t) (lambda (v) (lambda () (raise v)))])
                      (call-with-values thunk
                                        (lambda vs (lambda () (apply values vs)))))))))
  (cond
    [(sync/timeout seconds t) (outcome)]
    [else
     (kill-thread t)
     (fail (format "no result within ~a s" seconds))]))

(define-syntax-parse-rule (bounded-test name:expr body:expr ...+)
  (test-case name (within 60 (lambda () body ...))))
