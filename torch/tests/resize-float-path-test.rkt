#lang racket/base

(module+ test
  (require rackunit
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in "private/child.rkt" run-child)
           (only-in "private/within.rkt" bounded-test))

  (define-runtime-path resize-test "resize-test.rkt")
  (define-runtime-path transforms "../vision/transforms.rkt")

  (define float-path-program
    `(begin
       (require (only-in rackunit/log test-log)
                (only-in (submod (file ,(path->string transforms)) private)
                         native-uint8-resize?))
       (dynamic-require '(submod (file ,(path->string resize-test)) test) #f)
       (test-log #:display? #f #:exit? #t)
       (write native-uint8-resize?)))

  (bounded-test "resize's laws hold on the float path, under ATEN_CPU_CAPABILITY=default"
    (define-values (status native? err)
      (run-child float-path-program
                 #:environment (list (cons "ATEN_CPU_CAPABILITY" #"default"))))
    (check-equal? status 0 err)
    (check-false native? err)))
