#lang racket/base

(module+ test
  (require (only-in racket/file file->bytes)
           ;; whole-module: define-runtime-path needs phase-1 bindings
           ;; only-in strips
           racket/runtime-path
           (only-in rackunit check-equal? check-exn check-true test-case)
           (only-in "../main.rkt"
                    cpu-device select tensor->bytes tensor->list tensor-device
                    tensor-dtype tensor-shape)
           (only-in "../vision/image.rkt" decode-image read-image))

  (define-runtime-path fixtures "../vision/fixtures/images")
  (define (fixture name) (build-path fixtures name))

  (define (plane img c) (tensor->list (select img 0 c)))

  (define (drawn f)
    (for*/list ([y (in-range 23)] [x (in-range 37)]) (f x y)))

  (test-case "a PNG comes back [C H W] uint8, every pixel as drawn"
    (define img (read-image (fixture "gradient.png")))
    (check-equal? (tensor-shape img) '(3 23 37))
    (check-equal? (tensor-dtype img) 'uint8)
    (check-equal? (plane img 0) (drawn (lambda (x _) (modulo (* 7 x) 256))))
    (check-equal? (plane img 1) (drawn (lambda (_ y) (modulo (* 11 y) 256))))
    (check-equal? (plane img 2)
                  (drawn (lambda (x y) (modulo (+ (* 3 x) (* 5 y)) 256)))))

  (test-case "the stored channels by default, a requested count on demand"
    (define rgba (read-image (fixture "gradient-rgba.png")))
    (check-equal? (tensor-shape rgba) '(4 23 37))
    (check-equal? (plane rgba 3) (drawn (lambda (x y) (modulo (* x y) 256))))
    (define gray (read-image (fixture "gradient-gray.png")))
    (check-equal? (tensor-shape gray) '(1 23 37))
    (check-equal? (plane gray 0)
                  (drawn (lambda (x y) (modulo (+ (* 5 x) (* 3 y)) 256))))
    (define widened (read-image (fixture "gradient-gray.png") #:mode 'rgb))
    (check-equal? (tensor-shape widened) '(3 23 37))
    (check-equal? (plane widened 2) (plane gray 0))
    (define opaque (read-image (fixture "gradient.png") #:mode 'rgba))
    (check-equal? (plane opaque 3) (drawn (lambda (_x _y) 255)))
    (define (luma x y)
      (quotient (+ (* 77 (modulo (* 7 x) 256))
                   (* 150 (modulo (* 11 y) 256))
                   (* 29 (modulo (+ (* 3 x) (* 5 y)) 256)))
                256))
    (define gray-alpha (read-image (fixture "gradient.png") #:mode 'gray-alpha))
    (check-equal? (tensor-shape gray-alpha) '(2 23 37))
    (check-equal? (plane gray-alpha 0) (drawn luma)
                  "stb's luma, (77r + 150g + 29b) >> 8")
    (check-equal? (plane gray-alpha 1) (drawn (lambda (_x _y) 255)))
    (define reduced (read-image (fixture "gradient.png") #:mode 'gray))
    (check-equal? (tensor-shape reduced) '(1 23 37))
    (check-equal? (plane reduced 0) (drawn luma))
    (check-equal? (tensor-shape (read-image (fixture "palette.png")))
                  '(3 23 37)))

  (test-case "baseline, progressive and grayscale JPEGs decode"
    (for ([name (in-list '("smooth.jpg" "smooth-444.jpg"
                           "smooth-progressive.jpg"))])
      (check-equal? (tensor-shape (read-image (fixture name))) '(3 64 96)
                    name))
    (check-equal? (tensor-shape (read-image (fixture "smooth-gray.jpg")))
                  '(1 64 96))
    (check-equal? (tensor->bytes (read-image (fixture "smooth.jpg")))
                  (tensor->bytes (read-image (fixture "smooth-progressive.jpg")))
                  "one encoder's coefficients, two scan orders"))

  (test-case "decode-image reads bytes the way read-image reads a file"
    (define path (fixture "smooth.jpg"))
    (define from-bytes (decode-image (file->bytes path) #:device 'cpu))
    (check-equal? (tensor->bytes from-bytes) (tensor->bytes (read-image path)))
    (check-equal? (tensor-device from-bytes) (cpu-device)))

  (test-case "an image over the pixel limit is refused from its header"
    (define (be32 n) (integer->integer-bytes n 4 #f #t))
    (define header
      (bytes-append #"\211PNG\r\n\32\n" (be32 13) #"IHDR" (be32 16000)
                    (be32 16000) (bytes 8 0 0 0 0) (be32 0) (be32 0) #"IDAT"))
    (check-exn
     #rx"decode-image: .*16000 x 16000 pixels is over the 178956970-pixel limit"
     (lambda () (decode-image header))))

  (test-case "what is not an image is an error, what is not bytes a contract"
    (check-exn #rx"decode-image: .*cannot decode image: unknown image type"
               (lambda () (decode-image #"not an image at all")))
    (check-exn #rx"decode-image: .*cannot decode image"
               (lambda ()
                 (decode-image (subbytes (file->bytes (fixture "gradient.png"))
                                         0 40))))
    (check-exn #rx"read-image: .*cannot decode image"
               (lambda ()
                 (read-image (build-path fixtures 'up "cifar10-256.bin"))))
    (check-exn exn:fail:contract? (lambda () (decode-image #"")))
    (check-exn exn:fail:contract?
               (lambda () (read-image (fixture "smooth.jpg") #:mode 'cmyk)))))
