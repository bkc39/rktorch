#lang racket/base

(module+ main
  (require (submod "stepper.rkt" main))
  (call-with-output-file (getenv "BENCH_ENV_OUT")
    (lambda (out) (write (list (getenv "RKTORCH_MNIST_DIR") (getenv "RKTORCH_CIFAR10_DIR")) out))
    #:exists 'truncate))
