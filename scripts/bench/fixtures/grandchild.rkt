#lang racket/base

(require (only-in racket/file call-with-atomic-output-file)
         (only-in racket/os getpid))

(provide start-grandchild!)

(define (start-grandchild!)
  (define-values (grandchild _out in _err)
    (subprocess (current-output-port) #f (current-error-port)
                (find-executable-path "sleep") "120"))
  (close-output-port in)
  (call-with-atomic-output-file (getenv "BENCH_PID_FILE")
    (lambda (port _tmp) (write (list (getpid) (subprocess-pid grandchild)) port))))

(module+ main
  (start-grandchild!)
  (sleep 120))
