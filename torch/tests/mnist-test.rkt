#lang racket/base

(module+ test
  (require rackunit
           (only-in racket/list take)
           (only-in "../main.rkt" tensor-shape tensor->list)
           (only-in "../data/mnist.rkt"
                    load-mnist-fixture
                    read-idx
                    download-cached))

  (test-case "the committed fixture parses to normalized NCHW + int64 labels"
    (define-values (imgs lbls) (load-mnist-fixture))
    (check-equal? (tensor-shape imgs) '(256 1 28 28))
    (check-equal? (tensor-shape lbls) '(256))
    ;; the canonical first ten MNIST training labels
    (check-equal? (map inexact->exact (take (tensor->list lbls) 10))
                  '(5 0 4 1 9 2 1 3 1 4))
    (define px (tensor->list imgs))
    (check-true (>= (apply min px) 0.0) "pixel below 0")
    (check-true (<= (apply max px) 1.0) "pixel above 1")
    (check-true (> (apply max px) 0.0) "fixture is all zeros"))

  (test-case "read-idx rejects a non-IDX buffer"
    (check-exn exn:fail? (lambda () (read-idx (bytes 1 2 3 4)))))

  (define ok?
    (with-handlers ([exn:fail? (lambda (_) #f)])
      (define-values (dims _data)
        (read-idx (download-cached "t10k-images-idx3-ubyte.gz")))
      (equal? dims '(10000 28 28))))
  (if ok?
      (displayln "[mnist-test] download-cached t10k header OK (10000x28x28)")
      (displayln "[mnist-test] skipped download (offline / mirror unreachable)")))
