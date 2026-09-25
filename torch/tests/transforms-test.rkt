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
      (check-equal? (tensor-device (random-crop x)) (cuda-device))))

  (define (close? a expected [eps 1e-6])
    (define values (tensor->list a))
    (and (= (length values) (length expected))
         (for/and ([v (in-list values)] [e (in-list expected)])
           (< (abs (- v e)) eps))))

  (test-case "resize: an int is the short side and the long keeps the aspect"
    (check-equal? (tensor-shape (resize (randn 3 4 6) 2)) '(3 2 3))
    (check-equal? (tensor-shape (resize (randn 3 4 6) '(5 7))) '(3 5 7))
    (check-equal? (tensor-shape (resize (randn 2 3 6 4) 3)) '(2 3 4 3)
                  "4.5 rows truncate to 4"))

  (test-case "resize: the same size answers the image"
    (define x (randn 3 4 6))
    (check-true (same? (resize x '(4 6)) x))
    (check-true (same? (resize x 4 #:antialias? #f) x)))

  (test-case "resize: the two filters by hand"
    (define row (reshape (arange 4.0) 1 1 4))
    (check-true (close? (resize row '(1 2) #:antialias? #f) '(0.5 2.5))
                "plain bilinear averages the two inputs around each centre")
    (check-true (close? (resize row '(1 2)) '(5/7 16/7))
                "the antialiased triangle spans two inputs a side")
    (check-true (close? (resize (reshape (arange 9.0) 1 1 9) '(1 3))
                        '(1.25 4.0 6.75))
                "a whole-number factor puts the triangle's edge on a pixel")
    (define pair (reshape (arange 2.0) 1 1 2))
    (for ([antialias? (in-list '(#t #f))])
      (check-true (close? (resize pair '(1 4) #:antialias? antialias?)
                          '(0.0 0.25 0.75 1.0))
                  "upsampling clamps at the edges")))

  (test-case "resize: float64 and the half dtypes keep their dtype"
    (for ([dtype (in-list '(float64 float16 bfloat16))])
      (define out (resize (to-dtype (rand 3 8 8) dtype) 4))
      (check-equal? (tensor-dtype out) dtype)
      (check-equal? (tensor-shape out) '(3 4 4)))
    (check-exn exn:fail:contract?
               (lambda () (resize (to-dtype (rand 3 8 8) 'uint8) 4)))
    (check-exn exn:fail:contract? (lambda () (resize (rand 8 8) 4))))

  (test-case "center-crop: the offset rounds half to even, as Python's round"
    (define x (reshape (arange 25) 1 5 5))
    (check-equal? (tensor-shape (center-crop x 2)) '(1 2 2))
    (check-equal? (tensor->list (center-crop x 2)) '(12.0 13.0 17.0 18.0))
    (check-equal? (tensor->list (center-crop x '(4 1))) '(2.0 7.0 12.0 17.0))
    (check-equal? (tensor-shape (center-crop (rand 2 3 9 7) '(5 7)))
                  '(2 3 5 7))
    (check-exn exn:fail:contract? (lambda () (center-crop x 6)))
    (check-exn exn:fail:contract? (lambda () (center-crop x '(2 6)))))

  (test-case "normalize: per channel, one mean and one positive std each"
    (define x (reshape (tensor '(1.0 2.0 3.0 4.0)) 2 1 2))
    (check-true (close? (normalize x '(1 2) '(2 4)) '(0.0 0.5 0.25 0.5)))
    (check-exn exn:fail:contract? (lambda () (normalize x '(1) '(2))))
    (check-exn exn:fail:contract? (lambda () (normalize x '(1 2) '(2 0))))
    (define centred
      (imagenet-normalize (reshape (tensor imagenet-mean) 3 1 1)))
    (check-true (close? centred '(0.0 0.0 0.0)))
    (check-exn exn:fail:contract?
               (lambda () (imagenet-normalize (rand 1 4 4)))))

  (test-case "convert-image-dtype: torchvision's scaling each way"
    (define bytes-image (reshape (tensor #"\0\200\377") 1 1 3))
    (check-true (close? (convert-image-dtype bytes-image) '(0.0 128/255 1.0)))
    (define floats (reshape (tensor '(0.0 0.5 1.0)) 1 1 3))
    (check-equal? (tensor->list (convert-image-dtype floats 'uint8))
                  '(0 127 255))
    (check-eq? (convert-image-dtype floats 'float32) floats)
    (check-equal? (tensor-dtype (convert-image-dtype floats 'float64))
                  'float64))

  (test-case "imagenet-preprocess: the four steps in order, from bytes or floats"
    (define image (to-dtype (reshape (arange (* 3 300 400)) 3 300 400) 'uint8))
    (define by-hand
      (imagenet-normalize
       (center-crop (resize (convert-image-dtype image) 256) 224)))
    (check-true (same? (imagenet-preprocess image) by-hand))
    (check-true (same? (imagenet-preprocess (convert-image-dtype image))
                       by-hand))
    (check-equal? (tensor-shape (imagenet-preprocess
                                 (stack (list image image) 0)))
                  '(2 3 224 224))
    (check-exn exn:fail:contract?
               (lambda () (imagenet-preprocess (select image 0 0))))
    (check-exn exn:fail:contract?
               (lambda () (imagenet-preprocess (narrow image 0 0 1)))))

  (test-case "random-resized-crop: a square of the size, replayed by the seed"
    (define x (rand 3 30 40))
    (check-equal? (tensor-shape (random-resized-crop x 16)) '(3 16 16))
    (check-equal? (tensor-shape (random-resized-crop (rand 5 3 30 40) 8))
                  '(5 3 8 8))
    (check-true (same? (random-resized-crop x 16 #:generator (make-generator 3))
                       (random-resized-crop x 16 #:generator (make-generator 3)))))

  (test-case "random-resized-crop: the whole image when scale and ratio say so"
    (define x (rand 3 10 20))
    (check-true (same? (random-resized-crop x 8 #:scale '(1 1) #:ratio '(2 2))
                       (resize x '(8 8)))))

  (test-case "random-resized-crop: ten misses fall back to the central window"
    (define x (reshape (arange 81.0) 1 9 9))
    (check-true (same? (random-resized-crop x 3 #:scale '(1 1) #:ratio '(3 3))
                       (resize (narrow x 1 3 3) '(3 3)))
                "too wide for the image: its full width, a third as tall")
    (check-true (same? (random-resized-crop x 3 #:scale '(1 1)
                                            #:ratio '(1/3 1/3))
                       (resize (narrow x 2 3 3) '(3 3)))
                "too tall: its full height, a third as wide")
    (define big (rand 1 1000 1000))
    (check-true (same? (random-resized-crop big 3 #:scale '(1 1)
                                            #:ratio '(1/2 2)
                                            #:generator (make-generator 0))
                       (resize big '(3 3)))
                "the image's own ratio admissible: the whole image"))

  (test-case "random-resized-crop: what it refuses"
    (define x (rand 3 10 10))
    (check-exn exn:fail:contract?
               (lambda () (random-resized-crop x 4 #:scale '(0.5 0.1))))
    (check-exn exn:fail:contract?
               (lambda () (random-resized-crop x 4 #:scale '(0 1))))
    (check-exn exn:fail:contract?
               (lambda () (random-resized-crop x 4 #:ratio '(2 1))))
    (check-exn exn:fail:contract?
               (lambda () (random-resized-crop (to-dtype x 'uint8) 4)))))
