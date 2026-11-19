#lang racket/base

(module+ test
  (require rackunit
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in "../main.rkt" cpu-capability)
           (only-in "private/child.rkt" run-child)
           (only-in "private/within.rkt" bounded-test))

  (define-runtime-path main-module "../main.rkt")

  (define capability-program
    `(begin
       (require (only-in (file ,(path->string main-module)) cpu-capability))
       (write (cpu-capability))))

  (bounded-test "C: a lowercase symbol of ATen's, the same on every call"
    (define capability (cpu-capability))
    (check-true (symbol? capability))
    (check-equal? (symbol->string capability)
                  (string-downcase (symbol->string capability)))
    (check-not-false (memq capability '(default avx2 avx512 sve256 sve128 vsx zvector))
                     (format "~a" capability))
    (check-eq? (cpu-capability) capability))

  (bounded-test "C: ATEN_CPU_CAPABILITY=default at process start lowers it"
    (define-values (status capability err)
      (run-child capability-program
                 #:environment (list (cons "ATEN_CPU_CAPABILITY" #"default"))))
    (check-equal? status 0 err)
    (check-eq? capability 'default err)))
