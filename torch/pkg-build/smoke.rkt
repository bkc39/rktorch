#lang racket/base

(module+ test
  (require (only-in file/sha1 bytes->hex-string)
           (only-in racket/list first last)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in torch
                    accelerator-if-available cpu-device device-type mps-available?
                    tensor->bytes tensor->list)
           (only-in torch/audio/data load-audio save-audio)
           (only-in torch/foreign/raw/native-location native-library-directory)
           (only-in torch/private/util with-temporary-file)
           (only-in torch/scribblings/examples/04-mlp run-example)
           (only-in torch/vision/image read-image)
           (submod torch/tests/autograd-test test)
           (submod torch/tests/native-library-test test)
           (submod torch/tests/nn-test test)
           (submod torch/tests/tensor-ops-test test)
           (only-in threading ~>))

  (define-runtime-path flac-fixture "../audio/fixtures/sine-440-16k.flac")
  (define-runtime-path jpeg-fixture "../vision/fixtures/images/smooth.jpg")

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
      (check-equal? (tensor->list copied) (tensor->list samples))))

  (test-case "a 4:2:0 JPEG decodes through libjpeg-turbo to torchvision's pixels"
    (check-equal? (~> jpeg-fixture read-image tensor->bytes sha256-bytes
                      bytes->hex-string)
                  "11288f5546bfe6492d1b539f0eb17c5bc90810f537b0dcc968af8bca247ff6d7")))
