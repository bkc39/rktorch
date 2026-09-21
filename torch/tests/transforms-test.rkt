#lang racket/base

(module+ test
  (require rackunit
           "../main.rkt"
           "../vision/transforms.rkt")

  (define (same? a b)
    (equal? (tensor->list a) (tensor->list b)))

  (test-case "random-horizontal-flip: p = 0 keeps, p = 1 mirrors the width"
    (define x (reshape (arange 24) 2 3 2 2))
    (check-true (same? (random-horizontal-flip x #:p 0) x))
    (check-true (same? (random-horizontal-flip x #:p 1) (flip x 3)))
    (check-equal? (tensor-shape (random-horizontal-flip x)) '(2 3 2 2)))

  (test-case "random-horizontal-flip: a seeded generator replays its draws"
    (define x (randn 16 3 4 4))
    (define a (random-horizontal-flip x #:generator (make-generator 7)))
    (define b (random-horizontal-flip x #:generator (make-generator 7)))
    (check-true (same? a b))
    (define n 200)
    (define big (reshape (arange (* n 4)) n 1 2 2))
    (define mixed (random-horizontal-flip big #:generator (make-generator 1)))
    (define kept
      (for/sum ([i (in-range n)])
        (if (same? (select mixed 0 i) (select big 0 i)) 1 0)))
    (check-true (< 50 kept 150) "about half of 200 images stay"))

  (test-case "random-crop: padding 0 is the identity, otherwise a window"
    (define x (reshape (arange 32) 2 1 4 4))
    (check-true (same? (random-crop x #:padding 0) x))
    (define y (random-crop x #:padding 2 #:generator (make-generator 0)))
    (check-equal? (tensor-shape y) '(2 1 4 4))
    (check-equal? (tensor-dtype y) 'float32)
    ;; every output pixel is either padding or a pixel of its own image
    (for ([i (in-range 2)])
      (define values (tensor->list (select x 0 i)))
      (for ([v (in-list (tensor->list (select y 0 i)))])
        (check-not-false (or (= v 0.0) (memv v values)))))
    (check-true (same? y (random-crop x #:padding 2
                                      #:generator (make-generator 0)))))

  (test-case "random-crop: shifts differ across images and across batches"
    (define x (reshape (arange 64) 4 1 4 4))
    (define g (make-generator 3))
    (define a (random-crop x #:padding 4 #:generator g))
    (define b (random-crop x #:padding 4 #:generator g))
    (check-false (same? a b) "the generator stream advances per batch")
    (check-exn exn:fail:contract?
               (lambda () (random-crop (randn 3 4 4) #:padding 1))))

  (test-case "an empty batch passes through both transforms"
    (define x (zeros 0 3 4 4))
    (check-equal? (tensor-shape (random-horizontal-flip x)) '(0 3 4 4))
    (check-equal? (tensor-shape (random-crop x #:padding 2)) '(0 3 4 4)))

  (test-case "transforms keep the device of their input"
    (when (cuda-available?)
      (define x (to (randn 8 3 32 32) (cuda-device)))
      (check-equal? (tensor-device (random-horizontal-flip x)) (cuda-device))
      (check-equal? (tensor-device (random-crop x)) (cuda-device)))))
