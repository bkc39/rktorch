#lang racket/base

(module+ test
  (require (only-in racket/file file->bytes make-temporary-file)
           rackunit
           "../main.rkt"
           "../vision/ppm.rkt")

  (define (written image #:range [value-range '(0 1)])
    (define path (make-temporary-file "rkt-ppm-~a.ppm"))
    (write-ppm path image #:range value-range)
    (define bs (file->bytes path))
    (delete-file path)
    bs)

  (test-case "image-grid: make_grid's layout, padding around every image"
    ;; three 1x2 images of one channel each, two columns, padding 1
    (define images (reshape (add (arange 6) 1.0) 3 1 1 2))
    (define grid (image-grid images #:columns 2 #:padding 1 #:pad-value -1))
    (check-equal? (tensor-shape grid) '(3 5 7) "one channel becomes three")
    (define rows
      (for/list ([r (in-range 5)])
        (tensor->list (select (select grid 0 0) 0 r))))
    (check-equal? (car rows) '(-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0))
    (check-equal? (cadr rows) '(-1.0 1.0 2.0 -1.0 3.0 4.0 -1.0))
    (check-equal? (caddr rows) '(-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0))
    (check-equal? (cadddr rows) '(-1.0 5.0 6.0 -1.0 -1.0 -1.0 -1.0))
    (check-equal? (tensor->list (select (select grid 0 2) 0 1))
                  '(-1.0 1.0 2.0 -1.0 3.0 4.0 -1.0)
                  "the tripled channels agree")
    (check-equal? (tensor-shape (image-grid (randn 10 3 4 4))) '(3 14 50)
                  "ten images in eight columns: two rows")
    (check-equal? (tensor-shape (image-grid (randn 2 3 4 4) #:padding 0))
                  '(3 4 8))
    ;; a uint8 batch keeps its dtype, and full refuses a pad value outside
    ;; its range as contract blame rather than wrapping it
    (define bytes-batch (to-dtype (full 9.0 2 3 2 2) 'uint8))
    (define byte-grid (image-grid bytes-batch #:padding 1 #:pad-value 7))
    (check-equal? (tensor-dtype byte-grid) 'uint8)
    (check-equal? (tensor->list (select (select byte-grid 0 0) 0 0))
                  '(7 7 7 7 7 7 7))
    (check-exn #rx"^image-grid: contract violation"
               (lambda () (image-grid bytes-batch #:pad-value -1)))
    (check-exn #rx"uint8-fill-value"
               (lambda () (image-grid bytes-batch #:pad-value 256)))
    (define int-batch (to-dtype (full 9.0 2 3 2 2) 'int64))
    (check-exn #rx"^image-grid: contract violation"
               (lambda () (image-grid int-batch #:pad-value 0.5)))
    (check-exn #rx"int64-fill-value"
               (lambda () (image-grid int-batch #:pad-value 0.5))))

  (test-case "image-grid does not extend the caller's graph, as make_grid"
    (define x (mul (rand 3 3 2 2 #:requires-grad? #t) 1.0))
    (check-true (requires-grad? x))
    (check-false (requires-grad? (image-grid x)))
    ;; make_grid returns a three-channel singleton as a view, which torch
    ;; leaves requiring grad even under no_grad; a one-channel one is
    ;; built, so it does not
    (check-true (requires-grad? (image-grid (narrow x 0 0 1))))
    (check-false
     (requires-grad? (image-grid (narrow (narrow x 0 0 1) 1 0 1)))))

  (test-case "image-grid: one image comes back the way make_grid returns it"
    (define one (reshape (add (arange 12) 1.0) 1 3 2 2))
    (define g (image-grid one #:padding 2 #:pad-value -1))
    (check-equal? (tensor-shape g) '(3 2 2) "no border around a single image")
    (check-equal? (tensor->list g) (tensor->list (select one 0 0)))
    (define grey (reshape (add (arange 4) 1.0) 1 1 2 2))
    (define g1 (image-grid grey #:padding 2))
    (check-equal? (tensor-shape g1) '(3 2 2) "one channel still becomes three")
    (check-equal? (tensor->list (select g1 0 2)) '(1.0 2.0 3.0 4.0)))

  (test-case "write-ppm: a P6 header and one byte per channel, row-major"
    ;; red, green / blue, white
    (define image
      (tensor '(((1.0 0.0) (0.0 1.0))
                ((0.0 1.0) (0.0 1.0))
                ((0.0 0.0) (1.0 1.0)))))
    (check-equal? (written image)
                  (bytes-append #"P6\n2 2\n255\n"
                                (bytes 255 0 0 0 255 0 0 0 255 255 255 255)))
    (check-equal? (subbytes (written (sub (mul image 2.0) 1.0) #:range '(-1 1))
                            11)
                  (bytes 255 0 0 0 255 0 0 0 255 255 255 255)
                  "a [-1, 1] image quantizes the same")
    (check-equal? (subbytes (written (full 0.5 3 1 1)) 11) (bytes 128 128 128)
                  "half rounds to 128 like save_image")
    (check-equal? (subbytes (written (full 7.0 3 1 1)) 11) (bytes 255 255 255)
                  "values past the range clamp")
    (check-equal? (subbytes (written (to-dtype (full 9.0 3 1 2) 'uint8)) 11)
                  (bytes 9 9 9 9 9 9)
                  "uint8 passes through"))

  (test-case "image-grid and write-ppm: contracts on the batch, the image and the range"
    (check-exn #rx"non-empty-image-batch"
               (lambda () (image-grid (zeros 0 3 4 4))))
    (check-exn #rx"non-empty-image-batch"
               (lambda () (image-grid (zeros 3 4 4))))
    ;; a zero height or width makes a grid that is nothing but padding
    (check-exn #rx"non-empty-image-batch"
               (lambda () (image-grid (zeros 2 3 0 5))))
    (check-exn #rx"non-empty-image-batch"
               (lambda () (image-grid (zeros 2 3 5 0))))
    (check-exn #rx"non-empty-image-batch"
               (lambda () (image-grid (zeros 2 0 4 4))))
    (check-exn exn:fail:contract? (lambda () (written (zeros 2 2))))
    (check-exn exn:fail:contract? (lambda () (written (zeros 1 2 2))))
    (check-exn exn:fail:contract?
               (lambda () (written (zeros 3 2 2) #:range '(1 0))))
    ;; an infinite span makes the scale zero and writes every pixel black
    (check-exn #rx"value-range"
               (lambda () (written (zeros 3 2 2) #:range '(0 +inf.0))))
    (check-exn #rx"value-range"
               (lambda () (written (zeros 3 2 2) #:range '(-inf.0 1))))
    ;; finite endpoints whose span is not: the scale would come out zero
    (check-exn #rx"value-range"
               (lambda () (written (zeros 3 2 2) #:range '(-1e308 1e308))))
    ;; a span small enough that 255 over it is not a number either
    (check-exn #rx"value-range"
               (lambda () (written (zeros 3 2 2) #:range '(0 1e-307))))
    (check-exn #rx"expected: image"
               (lambda () (written (to-dtype (zeros 3 2 2) 'bool))))
    ;; an int64 image under the default range would quantize to white
    (check-exn #rx"expected: image"
               (lambda () (written (to-dtype (zeros 3 2 2) 'int64))))
    ;; a PPM header states a width and a height, and neither may be zero
    (check-exn #rx"expected: image" (lambda () (written (zeros 3 0 2))))
    (check-exn #rx"expected: image" (lambda () (written (zeros 3 2 0)))))

  (test-case "write-ppm quantizes off the graph, as save_image does"
    (define x (mul (rand 3 2 2 #:requires-grad? #t) 1.0))
    (check-true (requires-grad? x))
    (define path (make-temporary-file "rkt-grad-~a.ppm"))
    (write-ppm path x)
    (check-equal? (bytes-length (file->bytes path)) (+ 11 (* 3 2 2)))
    (delete-file path))

  (test-case "image-grid and write-ppm accept device tensors"
    (when (cuda-available?)
      (define x (to (rand 4 3 8 8) (cuda-device)))
      (define grid (image-grid x #:columns 2))
      (check-equal? (tensor-device grid) (cuda-device))
      ;; two rows of two 8x8 images with padding 2: 22 by 22, 13 header bytes
      (check-equal? (tensor-shape grid) '(3 22 22))
      (check-equal? (bytes-length (written grid)) (+ 13 (* 3 22 22))))))
