#lang racket/base
(require (only-in racket/list first last)
         rackunit
         (only-in torch/foreign/raw/native-location native-library-directory)
         (only-in torch/scribblings/examples/04-mlp run-example))

(printf "shim from ~a\n" (native-library-directory))
(define-values (losses _net device) (run-example))
(printf "device ~a, losses ~a\n" device losses)
(check < (last losses) (first losses) "the loss did not fall")
