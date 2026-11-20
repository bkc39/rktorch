#lang racket/base

(module+ test
  (require rackunit
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in (submod "../vision/transforms.rkt" private) native-uint8-resize?)
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

  (define float-path-reachable? (not (eq? (system-type 'arch) 'aarch64)))

  (cond
    [float-path-reachable?
     (bounded-test "resize's laws hold on the float path, under ATEN_CPU_CAPABILITY=default"
       (define-values (status native? err)
         (run-child float-path-program
                    #:environment (list (cons "ATEN_CPU_CAPABILITY" #"default"))))
       (check-equal? status 0 err)
       (check-false native? err))]
    [else
     (test-case "on aarch64 the uint8 resize is native at any capability, as torchvision's is"
       (check-true native-uint8-resize?))
     (displayln "[resize-float-path-test] skipped: aarch64 has no float path to force")]))
