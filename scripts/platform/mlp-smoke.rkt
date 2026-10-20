#lang racket/base
(require (only-in racket/list first last)
         (only-in rackunit check)
         (only-in torch accelerator-if-available cpu-device device-type mps-available?)
         (only-in torch/foreign/raw/native-location native-library-directory)
         (only-in torch/scribblings/examples/04-mlp run-example))

(define (train device)
  (define-values (losses _net used) (run-example device))
  (printf "device ~a, losses ~a\n" used losses)
  (check < (last losses) (first losses) (format "the loss did not fall on ~a" used)))

(printf "shim from ~a\n" (native-library-directory))
(printf "mps-available? ~a\n" (mps-available?))
(define picked (accelerator-if-available))
(train picked)
(unless (eq? (device-type picked) 'cpu)
  (train (cpu-device)))
