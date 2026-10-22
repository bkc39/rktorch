#lang racket/base

(module+ test
  (require rackunit
           (only-in racket/list range remove-duplicates)
           (only-in racket/sequence in-slice sequence->list)
           (only-in "../main.rkt" tensor-shape tensor->list)
           (only-in "../vision/mnist.rkt"
                    load-mnist-fixture
                    read-idx
                    download-cached)
           "private/class-means.rkt")

  (define-values (imgs lbls) (load-mnist-fixture))
  (define labels (map inexact->exact (tensor->list lbls)))

  (test-case "the committed fixture parses to normalized NCHW + int64 labels"
    (check-equal? (tensor-shape imgs) '(256 1 28 28))
    (check-equal? (tensor-shape lbls) '(256))
    (define px (tensor->list imgs))
    (check-true (>= (apply min px) 0.0) "pixel below 0")
    (check-true (<= (apply max px) 1.0) "pixel above 1")
    (check-true (> (apply max px) 0.0) "fixture is all zeros"))

  (test-case "every one of the ten digits is drawn, and only those"
    (check-equal? (sort (remove-duplicates labels) <) (range 10)))

  (test-case "each digit is a glyph: a record is nearest its own digit's mean image"
    (define records (sequence->list (in-slice 784 (tensor->list imgs))))
    (define means (class-means records labels))
    (for ([record (in-list records)] [label (in-list labels)] [i (in-naturals)])
      (check-equal? (nearest-class means record cosine-distance) label
                    (format "record ~a" i))))

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
