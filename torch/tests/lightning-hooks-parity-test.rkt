#lang racket/base

(module+ test
  (require (only-in racket/file delete-directory/files file->value
                    make-temporary-directory)
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in racket/system system*)
           rackunit
           (only-in "../train/private/hooks.rkt"
                    callback-hook? hook-log hook-name hook-named hooks
                    trainer-hook?)
           (only-in "private/lightning-names.rkt"
                    fixture lightning->rktorch log-permissions)
           (only-in "private/python-env.rkt"
                    python python-module-available? with-python-env))

  (define-runtime-path generator "python/lightning_hooks.py")
  (define-runtime-path fixture-path "lightning-hooks.rktd")

  (test-case "the hook table logs where Lightning's _FxValidator does"
    (check-equal? (log-permissions)
                  (for/hasheq ([h (in-list hooks)])
                    (values (hook-name h) (hook-log h)))))

  (test-case "every hook Lightning called is in the table, on its side"
    (for* ([events (in-hash-values (hash-ref fixture 'scenarios))]
           [e (in-list events)]
           [mapped (in-value (lightning->rktorch (car e)))]
           #:when mapped)
      (define h (hook-named (cadr mapped)))
      (check-true (and h ((if (eq? (car mapped) 'callback) callback-hook? trainer-hook?)
                          h))
                  (format "~a" (car e)))))

  (cond
    [(not (file-exists? generator))
     (displayln "[lightning-hooks-parity-test] skipped: no generator (installed package)")]
    [(not (python-module-available? "lightning"))
     (displayln "[lightning-hooks-parity-test] skipped: python3 has no lightning")]
    [else
     (test-case "the committed hook fixture is the generator's output"
       (displayln "[lightning-hooks-parity-test] running Lightning")
       (define root (make-temporary-directory "lightning-hooks-~a"))
       (define out (build-path root "lightning-hooks.rktd"))
       (define log (open-output-string))
       (define ok?
         (parameterize ([current-output-port log] [current-error-port log])
           (with-python-env (system* python "-I" generator out))))
       (check-true ok? (get-output-string log))
       (check-equal? (file->value out) (file->value fixture-path)
                     "torch/tests/lightning-hooks.rktd differs from the generator's output")
       (delete-directory/files root))]))
