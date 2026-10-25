#lang racket/base

(module+ test
  (require (only-in racket/file file->value)
           (only-in racket/list filter-map)
           (only-in racket/match match)
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in racket/system system*)
           rackunit
           (only-in "../private/util.rkt" with-temporary-directory)
           (only-in "../train/private/hooks.rkt"
                    callback-hook? closest-hook-name hook-form hook-indexed?
                    hook-log hook-name hook-named hooks trainer-hook?)
           (only-in "private/lightning-names.rkt"
                    fixture lightning->rktorch log-permissions)
           (only-in "private/python-env.rkt"
                    python python-module-available? with-python-env)
           (only-in "private/trainer-kit.rkt" batch-indexed?))

  (define-runtime-path generator "python/lightning_hooks.py")
  (define-runtime-path fixture-path "lightning-hooks.rktd")

  (define fixture-events
    (for*/list ([events (in-hash-values (hash-ref fixture 'scenarios))]
                [e (in-list events)])
      e))

  (test-case "the hook table logs where Lightning's _FxValidator does"
    (check-equal? (log-permissions)
                  (for/hasheq ([h (in-list hooks)])
                    (values (hook-name h) (hook-log h)))))

  (test-case "every hook Lightning called is in the table, on its side"
    (for ([e (in-list fixture-events)])
      (match (cons e (lightning->rktorch (car e)))
        [(cons _ #f) (void)]
        [(cons (cons lightning-name _) (list side name))
         (define h (hook-named name))
         (check-true (and h ((if (eq? side 'callback) callback-hook? trainer-hook?) h))
                     (format "~a" lightning-name))])))

  (test-case "a hook carries a batch index exactly where Lightning passes one"
    (for ([e (in-list fixture-events)])
      (match (cons e (lightning->rktorch (car e)))
        [(cons _ #f) (void)]
        [(cons (list lightning-name _ _ batch _) (list _ name))
         (define indexed? (and batch #t))
         (check-equal? (hook-indexed? (hook-named name)) indexed?
                       (format "the table, at ~a" lightning-name))
         (check-equal? (batch-indexed? name) indexed?
                       (format "the hook-order recorder, at ~a" lightning-name))])))

  (test-case "a misspelled hook name finds the closest definable hook"
    (check-equal? (closest-hook-name 'on-batch-end) 'on-train-batch-end)
    (check-equal? (closest-hook-name 'on-fit-strat) 'on-fit-start))

  (test-case "the forms that define hooks, in table order"
    (check-equal? (filter-map hook-form hooks)
                  '(define-callbacks define-optimizers define-training-step
                    define-validation-step define-test-step
                    define-predict-step)))

  (define (generated v) (hash-remove v 'torch))

  (cond
    [(not (file-exists? generator))
     (displayln "[lightning-hooks-parity-test] skipped: no generator (installed package)")]
    [(not (python-module-available? "lightning"))
     (displayln "[lightning-hooks-parity-test] skipped: python3 has no lightning")]
    [else
     (test-case "the committed hook fixture is the generator's output"
       (displayln "[lightning-hooks-parity-test] running Lightning")
       (with-temporary-directory (root #:template "lightning-hooks-~a")
         (define out (build-path root "lightning-hooks.rktd"))
         (define log (open-output-string))
         (define ok?
           (parameterize ([current-output-port log] [current-error-port log])
             (with-python-env (system* python "-I" generator out))))
         (check-true ok? (get-output-string log))
         (check-equal? (generated (file->value out)) (generated (file->value fixture-path))
                       "torch/tests/lightning-hooks.rktd differs from the generator's output")))]))
