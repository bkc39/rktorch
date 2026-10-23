#lang racket/base

(module+ main
  (require (only-in torch/foreign/raw/pressure-settings
                    native-collect-at-troughs native-collect-budget)
           (submod "stepper.rkt" main))
  (call-with-output-file (getenv "BENCH_ENV_OUT")
    (lambda (out) (write (list (native-collect-at-troughs) (native-collect-budget)) out))
    #:exists 'truncate))
