#lang racket/base

(require (only-in racket/function const)
         (only-in rackunit fail))

(provide within)

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
