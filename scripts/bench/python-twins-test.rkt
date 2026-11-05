#lang racket/base

(module+ test
  (require (only-in racket/port with-output-to-string)
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in racket/system system*)
           rackunit
           (only-in "system-tools.rkt" command-output))

  (define-runtime-path twins-test "python/twins_test.py")

  (cond
    [(not (command-output "python3" "-c" "import numpy, torch, torchvision"))
     (displayln "[python-twins-test] skipped: no python3 with torch and torchvision")]
    [else
     (test-case "the PyTorch twins' own unit tests pass"
       (define passed? #f)
       (define log
         (with-output-to-string
           (lambda ()
             (parameterize ([current-error-port (current-output-port)])
               (set! passed? (system* (find-executable-path "python3") (path->string twins-test)))))))
       (check-true passed? log))]))
