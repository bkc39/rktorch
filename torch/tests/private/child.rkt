#lang racket/base

(require (only-in compiler/find-exe find-exe)
         (only-in racket/dict in-dict)
         (only-in racket/port copy-port))

(provide run-child)

(define (run-child program #:environment [changes '()] #:within [seconds #f])
  (define env (environment-variables-copy (current-environment-variables)))
  (for ([(name value) (in-dict changes)])
    (environment-variables-set! env (string->bytes/utf-8 name) value))
  (define-values (sp out in err)
    (parameterize ([current-environment-variables env])
      (subprocess #f #f #f (find-exe) "-e" (format "~s" program))))
  (close-output-port in)
  (define stdout (open-output-string))
  (define stderr (open-output-string))
  (define drains (list (drain-into stdout out) (drain-into stderr err)))
  (define status
    (cond
      [(sync/timeout seconds sp) (subprocess-status sp)]
      [else (subprocess-kill sp #t) #f]))
  (define drained (+ (current-inexact-monotonic-milliseconds) 5000))
  (for ([t (in-list drains)])
    (sync/timeout (max 0 (/ (- drained (current-inexact-monotonic-milliseconds)) 1000.0)) t))
  (close-input-port out)
  (close-input-port err)
  (for-each thread-wait drains)
  (values status
          (read (open-input-string (get-output-string stdout)))
          (get-output-string stderr)))

(define (drain-into sink port)
  (thread (lambda () (with-handlers ([exn:fail? void]) (copy-port port sink)))))

(module+ test
  (require rackunit)

  (test-case "a child's status, the datum it wrote and its stderr come back"
    (define-values (status answer stderr)
      (run-child '(begin (write (list 1 (getenv "CHILD_PROBE"))) (eprintf "said") (exit 3))
                 #:environment (list (cons "CHILD_PROBE" #"set"))))
    (check-equal? status 3)
    (check-equal? answer '(1 "set"))
    (check-equal? stderr "said"))

  (test-case "a child past its bound is killed and has no status"
    (define start (current-inexact-monotonic-milliseconds))
    (define-values (status answer _stderr) (run-child '(sync never-evt) #:within 1))
    (check-false status)
    (check-true (eof-object? answer))
    (check-true (< (- (current-inexact-monotonic-milliseconds) start) 10000)))

  (define sleep-exe (path->string (find-executable-path "sleep")))

  (define (holding-output seconds)
    `(let-values ([(_sp _out in _err)
                   (subprocess (current-output-port) #f (current-error-port)
                               ,sleep-exe ,(number->string seconds))])
       (close-output-port in)))

  (test-case "output a child leaves open past its exit is drained for 5 s in all, not per port"
    (define start (current-inexact-monotonic-milliseconds))
    (define-values (status _answer _stderr) (run-child (holding-output 15) #:within 30))
    (check-equal? status 0)
    (check-true (< (- (current-inexact-monotonic-milliseconds) start) 8000)))

  (test-case "a child's answer survives a grandchild holding its output for 3 s"
    (define-values (status answer stderr)
      (run-child `(begin (write '(answered)) (flush-output) ,(holding-output 3)) #:within 30))
    (check-equal? status 0 stderr)
    (check-equal? answer '(answered)))

  (test-case "an answer written before a grandchild holds the output past the cap comes back"
    (define noise (open-output-string))
    (define-values (status answer stderr)
      (parameterize ([current-error-port noise])
        (run-child `(begin (write '(answered)) (flush-output) ,(holding-output 6)) #:within 30)))
    (check-equal? status 0 stderr)
    (check-equal? answer '(answered))
    (check-equal? (get-output-string noise) "")))
