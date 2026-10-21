#lang racket/base

(module+ test
  (require (only-in racket/file delete-directory/files file->bytes
                    make-temporary-directory)
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in racket/system system*)
           rackunit
           (only-in "private/python-env.rkt" python))

  (define-runtime-path generator "../../scripts/gen-dataset-fixtures.py")
  (define-runtime-path fixtures "../vision/fixtures")

  (define generated
    '("cifar10-256.bin" "mnist-256-images-idx3-ubyte" "mnist-256-labels-idx1-ubyte"))

  (cond
    [(not (file-exists? generator))
     (displayln "[dataset-fixtures-parity-test] skipped: no generator (installed package)")]
    [(not python)
     (test-case "python3 is on PATH wherever the generator is"
       (fail "scripts/gen-dataset-fixtures.py is present but python3 is not on PATH"))]
    [else
     (test-case "the committed fixtures are the generator's output, byte for byte"
       (displayln "[dataset-fixtures-parity-test] running the generator")
       (define out (make-temporary-directory "dataset-fixtures-~a"))
       (check-true (system* python "-I" generator out))
       (for ([name (in-list generated)])
         (check-true (equal? (file->bytes (build-path out name))
                             (file->bytes (build-path fixtures name)))
                     (format "~a differs from the generator's output" name)))
       (delete-directory/files out))]))
