#lang racket/base

(require (only-in racket/contract/base
                  ->* and/c flat-named-contract list/c)
         (only-in "../foreign.rkt"
                  add cat clamp copy! full mul narrow permute select sub
                  tensor-device tensor-dtype tensor-shape tensor->vector
                  tensor? to-dtype)
         (only-in "../foreign/contracts.rkt" image-batch/c)
         (only-in "../private/contract.rkt" define/contract-out))

(define image/c
  (flat-named-contract
   'image
   (lambda (x)
     (and (tensor? x)
          (let ([dims (tensor-shape x)])
            (and (= 3 (length dims)) (= 3 (car dims))))))))

(define value-range/c
  (and/c (list/c real? real?) (lambda (r) (< (car r) (cadr r)))))

;; torchvision.utils.make_grid's layout, so a twin can pin it: columns
;; across, padding around every image, one channel tripled to three
(define/contract-out (image-grid images ;; noqa
                                 #:columns [columns 8]
                                 #:padding [padding 2]
                                 #:pad-value [pad-value 0])
  (->* [image-batch/c]
       [#:columns exact-positive-integer?
        #:padding exact-nonnegative-integer?
        #:pad-value real?]
       tensor?)
  (define dims (tensor-shape images))
  (define n (car dims))
  (define c (cadr dims))
  (define h (caddr dims))
  (define w (cadddr dims))
  (define rgb (if (= c 1) (cat (list images images images) 1) images))
  (define cols (min columns n))
  (define rows (quotient (+ n cols -1) cols))
  (define grid
    (full pad-value (if (= c 1) 3 c)
          (+ (* rows (+ h padding)) padding)
          (+ (* cols (+ w padding)) padding)
          #:device (tensor-device images)
          #:dtype (tensor-dtype images)))
  (for ([i (in-range n)])
    (define top (+ padding (* (quotient i cols) (+ h padding))))
    (define left (+ padding (* (remainder i cols) (+ w padding))))
    (copy! (narrow (narrow grid 1 top h) 2 left w) (select rgb 0 i)))
  grid)

;; save_image's quantization: scale to [0, 255], add a half, clamp,
;; truncate; a uint8 image is written as it is
(define/contract-out (write-ppm path image #:range [value-range '(0 1)]) ;; noqa
  (->* [path-string? image/c] [#:range value-range/c] void?)
  (define dims (tensor-shape image))
  (define lo (car value-range))
  (define hi (cadr value-range))
  (define pixels
    (tensor->vector
     (permute (if (eq? (tensor-dtype image) 'uint8)
                  image
                  (to-dtype (clamp (add (mul (sub image lo) (/ 255.0 (- hi lo)))
                                        0.5)
                                   #:min 0 #:max 255)
                            'uint8))
              1 2 0)))
  (call-with-output-file path
    #:exists 'truncate
    (lambda (out)
      (write-bytes
       (string->bytes/latin-1 (format "P6\n~a ~a\n255\n" (caddr dims) (cadr dims)))
       out)
      (write-bytes pixels out)
      (void))))
