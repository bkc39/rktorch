#lang racket/base

(module+ test
  (require (only-in racket/list first last)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in "../audio/data.rkt" load-audio save-audio)
           (only-in "../foreign/raw/native-location.rkt" native-library-directory)
           (only-in "../main.rkt"
                    accelerator-if-available cpu-device device-type mps-available?
                    tensor->list)
           (only-in "../private/util.rkt" with-temporary-file)
           (only-in "../scribblings/examples/04-mlp.rkt" run-example)
           (submod "../tests/autograd-test.rkt" test)
           (submod "../tests/native-library-test.rkt" test)
           (submod "../tests/nn-test.rkt" test)
           (submod "../tests/tensor-ops-test.rkt" test))

  (define-runtime-path flac-fixture "../audio/fixtures/sine-440-16k.flac")

  (printf "shim from ~a\n" (native-library-directory))
  (printf "mps-available? ~a\n" (mps-available?))

  (define (train device)
    (define-values (losses _net used) (run-example device))
    (printf "04-mlp on ~a: losses ~a\n" used losses)
    (check < (last losses) (first losses) (format "the loss did not fall on ~a" used)))

  (test-case "04-mlp trains on the device it picks, and on the CPU when that is an accelerator"
    (define picked (accelerator-if-available))
    (train picked)
    (unless (eq? (device-type picked) 'cpu)
      (train (cpu-device))))

  (test-case "a FLAC file round-trips through libsndfile"
    (define-values (samples rate) (load-audio flac-fixture))
    (with-temporary-file (copy #:template "round-trip-~a.flac")
      (save-audio copy samples rate)
      (define-values (copied copied-rate) (load-audio copy))
      (check-equal? copied-rate rate)
      (check-equal? (tensor->list copied) (tensor->list samples)))))
