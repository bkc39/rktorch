#lang racket/base

(module+ test
  ;; whole-module on purpose: the expansion needs bindings only-in would strip
  (require racket/runtime-path
           rackunit
           (only-in "../foreign/raw/pressure-settings.rkt"
                    memory-fraction-from-env memory-limit-from-env))

  (define-runtime-path settings-module "../foreign/raw/pressure-settings.rkt")

  (define mib (* 1024 1024))

  (define (with-env name supplied thunk)
    (parameterize ([current-environment-variables (make-environment-variables)])
      (when supplied
        (putenv name supplied))
      (thunk)))

  (define (fraction-from supplied)
    (with-env "RKTORCH_MEMORY_FRACTION" supplied memory-fraction-from-env))

  (define (limit-from supplied)
    (with-env "RKTORCH_MEMORY_LIMIT" supplied memory-limit-from-env))

  (define (check-rejected read name supplied expected)
    (define message (format "~a: expected ~a, given ~s" name expected supplied))
    (check-exn (lambda (e)
                 (and (exn:fail:user? e) (equal? (exn-message e) message)))
               (lambda () (read supplied))
               message))

  (test-case "unset, both settings keep their defaults"
    (check-equal? (fraction-from #f) 4/5)
    (check-equal? (limit-from #f) #f))

  (test-case "RKTORCH_MEMORY_FRACTION takes a share of capacity, decimals exact"
    (check-equal? (fraction-from "1/2") 1/2)
    (check-equal? (fraction-from "0.5") 1/2)
    (check-equal? (fraction-from "0.8") 4/5)
    (check-equal? (fraction-from "1") 1))

  (test-case "RKTORCH_MEMORY_LIMIT is in MiB, rounded down to whole bytes"
    (check-equal? (limit-from "6062") (* 6062 mib))
    (check-equal? (limit-from "1.5") (* 3/2 mib))
    (check-equal? (limit-from "0.1") 104857))

  (test-case "a malformed setting fails, naming the variable and the value"
    (for ([supplied (in-list '("0" "1.5" "-0.5" "" "half" "+nan.0" "+inf.0"
                               "1+2i"))])
      (check-rejected fraction-from "RKTORCH_MEMORY_FRACTION" supplied
                      "a number in (0, 1]"))
    (for ([supplied (in-list '("0" "-1" "6 GiB" "+inf.0" "1e-9"))])
      (check-rejected limit-from "RKTORCH_MEMORY_LIMIT" supplied
                      "a positive number of MiB")))

  ;; a fresh namespace instantiates the module again, so it reads the
  ;; environment as a new process would
  (define (initial-values env)
    (parameterize ([current-environment-variables (make-environment-variables)]
                   [current-namespace (make-base-empty-namespace)])
      (for ([(name supplied) (in-hash env)])
        (putenv name supplied))
      (list ((dynamic-require settings-module 'native-memory-fraction))
            ((dynamic-require settings-module 'native-memory-limit)))))

  (test-case "the parameters start from the environment"
    (check-equal? (initial-values (hash)) (list 4/5 #f))
    (check-equal? (initial-values (hash "RKTORCH_MEMORY_FRACTION" "2/3"
                                        "RKTORCH_MEMORY_LIMIT" "6062"))
                  (list 2/3 (* 6062 mib)))
    (check-exn #rx"^RKTORCH_MEMORY_FRACTION: expected a number in \\(0, 1\\], given \"4/5 \"$"
               (lambda () (initial-values (hash "RKTORCH_MEMORY_FRACTION" "4/5 "))))))
