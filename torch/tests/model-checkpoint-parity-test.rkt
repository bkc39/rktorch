#lang racket/base

(module+ test
  (require (only-in racket/file file->value)
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in racket/system system*)
           rackunit
           (only-in "../private/util.rkt" with-temporary-directory)
           (only-in "private/python-env.rkt" python python-module-available? with-python-env))

  (define-runtime-path generator "python/lightning_checkpoint.py")
  (define-runtime-path fixture-path "lightning-checkpoint.rktd")

  (cond
    [(not (file-exists? generator))
     (displayln "[model-checkpoint-parity-test] skipped: no generator (installed package)")]
    [(not (python-module-available? "lightning"))
     (displayln "[model-checkpoint-parity-test] skipped: python3 has no lightning")]
    [else
     (test-case "T3-L25 (PS): the committed ModelCheckpoint fixture is the generator's output"
       (with-temporary-directory (root #:template "lightning-checkpoint-~a")
         (define out (build-path root "lightning-checkpoint.rktd"))
         (define log (open-output-string))
         (define ok?
           (parameterize ([current-output-port log] [current-error-port log])
             (with-python-env (system* python "-I" generator (path->string out)))))
         (check-true ok? (get-output-string log))
         (check-equal? (file->value out) (file->value fixture-path)
                       "torch/tests/lightning-checkpoint.rktd differs from the generator's output")))]))
