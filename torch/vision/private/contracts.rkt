#lang racket/base

(require (only-in racket/contract/base
                  >=/c and/c flat-contract-predicate flat-named-contract list/c
                  or/c real-in)
         (only-in racket/match match-lambda match*)
         (only-in threading ~>)
         (only-in "../../foreign.rkt" device dtype generator? shape tensor?)
         (only-in "../../foreign/contracts.rkt" image-batch/c rgb-image-batch/c)
         (only-in "../../foreign/generator.rkt" generator-on?)
         (only-in "../../foreign/ops.rkt" any-float-dtype/c)
         (only-in "../../private/contract.rkt" supplied-or))

(provide batch-for/c
         crop-boxes/c
         default-ratio
         default-scale
         float-batch/c
         generator/c
         image-size/c
         jitter-strengths/c
         nonnegative/c
         ordered-pair/c
         per-sample-mask/c
         per-sample/c
         ratio/c
         rgb-float-batch/c
         rgb-image-batch-u8-or-float/c
         rgb-triple/c
         scale/c)

(define float-dtype? (flat-contract-predicate any-float-dtype/c))

(define (float-tensor? x)
  (and (tensor? x) (float-dtype? (dtype x))))

(define (same-device? a b)
  (equal? (device a) (device b)))

(define nonnegative/c (and/c real? (>=/c 0)))

(define (ordered-pair/c element/c)
  (flat-named-contract
   'ascending-pair
   (and/c (list/c element/c element/c) (match-lambda [(list lo hi) (<= lo hi)]))))

(define scale/c (~> (real-in 0 1) (and/c positive?) ordered-pair/c))
(define ratio/c (ordered-pair/c (and/c real? positive?)))
(define default-scale '(0.08 1.0))
(define default-ratio '(3/4 4/3))

(define image-size/c
  (flat-named-contract
   'image-size
   (or/c exact-positive-integer?
         (list/c exact-positive-integer? exact-positive-integer?))))

(define float-batch/c
  (flat-named-contract 'float-image-batch (and/c image-batch/c float-tensor?)))

(define rgb-float-batch/c
  (flat-named-contract 'rgb-float-image-batch
                       (and/c rgb-image-batch/c float-tensor?)))

(define rgb-image-batch-u8-or-float/c
  (flat-named-contract
   'rgb-image-batch
   (and/c rgb-image-batch/c
          (lambda (x) (or (eq? (dtype x) 'uint8) (float-tensor? x))))))

(define generator/c (or/c generator? #f))

(define (batch-for/c batch/c generator)
  (and/c batch/c
         (flat-named-contract
          'on-the-generator-device
          (lambda (x) (generator-on? (supplied-or generator) (device x))))))

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

(define jitter-strengths/c
  (flat-named-contract
   'jitter-strengths
   (list/c nonnegative/c nonnegative/c nonnegative/c (real-in 0 0.5))))

(define (rgb-triple/c [element/c real?])
  (flat-named-contract 'rgb-triple (list/c element/c element/c element/c)))
