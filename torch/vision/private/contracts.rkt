#lang racket/base

(require (only-in racket/contract/base
                  >=/c and/c flat-named-contract list/c or/c real-in)
         (only-in racket/match match match*)
         (only-in "../../foreign.rkt" device dtype generator? shape tensor?)
         (only-in "../../foreign/generator.rkt" generator-on?))

(provide crop-boxes/c
         float-batch/c
         generator-for/c
         image-size/c
         jitter-strengths/c
         ordered-pair/c
         per-sample-mask/c
         per-sample/c
         rgb-float-batch/c
         rgb-image-batch-u8-or-float/c
         rgb-triple/c)

(define float-dtypes '(float16 bfloat16 float32 float64))

(define (float-tensor? x)
  (and (tensor? x) (memq (dtype x) float-dtypes) #t))

(define (same-device? a b)
  (equal? (device a) (device b)))

(define (ordered-pair/c element/c)
  (flat-named-contract
   'ascending-pair
   (and/c (list/c element/c element/c)
          (lambda (pair) (<= (car pair) (cadr pair))))))

(define image-size/c
  (flat-named-contract
   'image-size
   (or/c exact-positive-integer?
         (list/c exact-positive-integer? exact-positive-integer?))))

(define float-batch/c
  (flat-named-contract
   'float-image-batch
   (lambda (x) (and (float-tensor? x) (= 4 (length (shape x)))))))

(define (three-channels? x)
  (match (shape x)
    [(list _ 3 _ _) #t]
    [_ #f]))

(define rgb-float-batch/c
  (flat-named-contract
   'rgb-float-image-batch
   (lambda (x) (and (float-tensor? x) (three-channels? x)))))

(define rgb-image-batch-u8-or-float/c
  (flat-named-contract
   'rgb-image-batch
   (lambda (x)
     (and (tensor? x)
          (or (eq? (dtype x) 'uint8) (float-tensor? x))
          (three-channels? x)))))

(define (rows-of? t x row-shape)
  (match* ((shape t) (shape x))
    [((cons n rest) (cons n _)) (equal? rest row-shape)]
    [(_ _) #f]))

(define (crop-boxes/c x)
  (flat-named-contract
   'crop-boxes
   (lambda (boxes)
     (and (float-tensor? boxes)
          (rows-of? boxes x '(4))
          (same-device? boxes x)))))

(define (per-sample/c x [real/c real?])
  (flat-named-contract
   'per-sample-factor
   (or/c real/c
         (lambda (f)
           (and (float-tensor? f) (rows-of? f x '()) (same-device? f x))))))

(define (per-sample-mask/c x)
  (flat-named-contract
   'per-sample-mask
   (lambda (m)
     (and (tensor? m)
          (eq? (dtype m) 'bool)
          (rows-of? m x '())
          (same-device? m x)))))

(define (generator-for/c x)
  (flat-named-contract
   'generator-on-the-batch-device
   (lambda (g)
     (and (or (not g) (generator? g))
          (generator-on? g (device x))))))

(define jitter-strengths/c
  (flat-named-contract
   'jitter-strengths
   (list/c (and/c real? (>=/c 0))
           (and/c real? (>=/c 0))
           (and/c real? (>=/c 0))
           (real-in 0 0.5))))

(define (rgb-triple/c [element/c real?])
  (flat-named-contract 'rgb-triple (list/c element/c element/c element/c)))
