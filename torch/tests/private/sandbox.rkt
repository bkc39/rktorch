#lang racket/base

(require (only-in racket/sandbox kill-evaluator set-eval-limits)
         (only-in scribble/example make-base-eval)
         ;; whole-module: the pattern's syntax classes live at phase 1 and
         ;; only-in would strip them
         syntax/parse/define)

(provide call-with-fresh-eval
         with-fresh-eval)

;; make-base-eval installs the trusted sandbox configuration, which resets
;; sandbox-eval-limits to #f, so the limit is set on the evaluator itself.
(define (call-with-fresh-eval proc
                              #:prelude [prelude '()]
                              #:time-limit [secs 60])
  (define ev (apply make-base-eval prelude))
  (set-eval-limits ev secs #f)
  (dynamic-wind void
                (lambda () (proc ev))
                (lambda () (kill-evaluator ev))))

(define-syntax-parse-rule (with-fresh-eval (ev:id option ...) body:expr ...+)
  (call-with-fresh-eval (lambda (ev) body ...) option ...)) ;; noqa

(module+ test
  (require (only-in racket/sandbox exn:fail:resource?)
           rackunit)

  (test-case "an evaluation past the time limit is stopped"
    (with-fresh-eval (ev #:time-limit 1)
      (check-exn exn:fail:resource?
                 (lambda () (ev '(begin (sleep 5) 'finished)))))))
