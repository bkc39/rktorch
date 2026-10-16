#lang racket/base

(module+ main
  (call-with-output-file "rktorch-steps.json" #:exists 'truncate
    (lambda (port) (write-string "{\"steps\":[[1" port)))
  (exit 1))
