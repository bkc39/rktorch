#lang racket/base

(require (only-in racket/contract/base
                  ->* ->i and/c flat-named-contract list/c unsupplied-arg?)
         (only-in "../foreign.rkt"
                  add clamp copy! fill-value/c full mul narrow permute select
                  sub tensor-device tensor-dtype tensor-shape tensor->vector
                  tensor? to-dtype with-no-grad)
         (only-in "../foreign/contracts.rkt" image-batch/c)
         (only-in "../private/contract.rkt" define/contract-out))

(define ppm-dtypes '(float16 bfloat16 float32 float64 uint8))

(define image/c
  (flat-named-contract
   'image
   (lambda (x)
     (and (tensor? x)
          (and (memq (tensor-dtype x) ppm-dtypes) #t)
          (let ([dims (tensor-shape x)])
            (and (= 3 (length dims))
                 (= 3 (car dims))
                 (andmap positive? dims)))))))

;; the endpoints cross to ATen as doubles, so the span and scale checked
;; here are the ones the arithmetic will use, not the exact originals
(define (quantizes? r)
  (define lo (exact->inexact (car r)))
  (define hi (exact->inexact (cadr r)))
  (define span (- hi lo))
  (and (= lo (car r))
       (= hi (cadr r))
       (< lo hi)
       (rational? span)
       (positive? span)
       (let ([scale (/ 255.0 span)])
         (and (rational? scale) (positive? scale)))))

;; ATen applies the scale in float32 arithmetic for a float32 image and
;; for the half dtypes, whose kernels compute in float32: a span small
;; enough to make it overflow there writes every pixel white
(define (scale-fits? value-range dtype)
  (define span (- (exact->inexact (cadr value-range))
                  (exact->inexact (car value-range))))
  (or (memq dtype '(float64 uint8))
      (<= (/ 255.0 span) 3.4028234663852886e38)))

(define value-range/c
  (flat-named-contract
   'value-range
   (and/c (list/c rational? rational?) quantizes?)))

(define non-empty-image-batch/c
  (flat-named-contract
   'non-empty-image-batch
   (and/c image-batch/c (lambda (x) (andmap positive? (tensor-shape x))))))

(define/contract-out (image-grid images ;; noqa
                                 #:columns [columns 8]
                                 #:padding [padding 2]
                                 #:pad-value [pad-value 0])
  (->i ([images non-empty-image-batch/c])
       (#:columns [columns exact-positive-integer?]
        #:padding [padding exact-nonnegative-integer?]
        #:pad-value [pad-value (images) (fill-value/c (tensor-dtype images))])
       [result tensor?])
  (define dims (tensor-shape images))
  (define n (car dims))
  (define c (cadr dims))
  (define h (caddr dims))
  (define w (cadddr dims))
  ;; make_grid is a display helper and carries @torch.no_grad(); without
  ;; it every copy! would extend the caller's graph into the grid
  (with-no-grad
    (cond
      [(= n 1) (one-image (select images 0 0) c h w)]
      [else (grid-of images n c h w columns padding pad-value)])))

(define (one-image image c h w)
  (cond
    [(= c 1)
     (define out (full 0.0 3 h w
                       #:device (tensor-device image)
                       #:dtype (tensor-dtype image)))
     (copy! out image)
     out]
    [else image]))

(define (grid-of images n c h w columns padding pad-value)
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
    ;; copy! broadcasts, so a one-channel tile fills all three channels
    (copy! (narrow (narrow grid 1 top h) 2 left w) (select images 0 i)))
  grid)

(define/contract-out (write-ppm path image #:range [value-range '(0 1)]) ;; noqa
  (->i ([path path-string?] [image image/c])
       (#:range [value-range value-range/c])
       #:pre/name (image value-range)
       "the range's scale must be a number in the image's dtype"
       (or (unsupplied-arg? value-range)
           (scale-fits? value-range (tensor-dtype image)))
       [_ void?])
  (define dims (tensor-shape image))
  (define lo (car value-range))
  (define hi (cadr value-range))
  (define pixels
    (with-no-grad
      (tensor->vector
       (permute (if (eq? (tensor-dtype image) 'uint8)
                    image
                    (to-dtype (clamp (add (mul (sub image lo)
                                               (/ 255.0 (- hi lo)))
                                          0.5)
                                     #:min 0 #:max 255)
                              'uint8))
                1 2 0))))
  (call-with-output-file path
    #:exists 'truncate
    (lambda (out)
      (write-bytes
       (string->bytes/latin-1 (format "P6\n~a ~a\n255\n" (caddr dims) (cadr dims)))
       out)
      (write-bytes pixels out)
      (void))))
