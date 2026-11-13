#lang racket/base

(require (only-in compiler/find-exe find-exe)
         (only-in racket/dict in-dict)
         (only-in racket/system system*/exit-code))

(provide run-child)

(define (run-child program #:environment [changes '()])
  (define env (environment-variables-copy (current-environment-variables)))
  (for ([(name value) (in-dict changes)])
    (environment-variables-set! env (string->bytes/utf-8 name) value))
  (define out (open-output-string))
  (define err (open-output-string))
  (define status
    (parameterize ([current-environment-variables env]
                   [current-output-port out]
                   [current-error-port err])
      (system*/exit-code (find-exe) "-e" (format "~s" program))))
  (values status
          (read (open-input-string (get-output-string out)))
          (get-output-string err)))
