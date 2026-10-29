#lang racket/base

(require (only-in threading ~>)
         (only-in "../../foreign.rkt"
                  * + - / <= >= cat clamp device dtype eq full narrow reshape
                  tensor? to-dtype where)
         (only-in "../../generated.rkt" floor-tensor fmod-scalar mean-dim))

(provide blend
         brightened
         column
         contrast-mean
         hue-turned
         luma
         solarized)

(define (column x f)
  (cond
    [(tensor? f) (~> f (reshape -1 1 1 1) (to-dtype (dtype x)))]
    [else (full (exact->inexact f) 1 1 1 1 #:dtype (dtype x) #:device (device x))]))

(define (channel x k)
  (narrow x 1 k 1))

(define (luma x)
  (+ (* 0.2989 (channel x 0)) (* 0.587 (channel x 1)) (* 0.114 (channel x 2))))

(define (unit x)
  (clamp x #:min 0.0 #:max 1.0))

(define (blend x other factor)
  (unit (+ (* factor x) (* (- 1.0 factor) other))))

(define (brightened x factor)
  (unit (* x factor)))

(define (solarized x threshold)
  (where (>= x threshold) (- 1.0 x) x))

(define (larger a b)
  (where (>= a b) a b))

(define (smaller a b)
  (where (<= a b) a b))

(define (rgb->hsv x)
  (define-values (r g b) (values (channel x 0) (channel x 1) (channel x 2)))
  (define maxc (larger (larger r g) b))
  (define minc (smaller (smaller r g) b))
  (define flat (eq maxc minc))
  (define span (- maxc minc))
  (define s (/ span (where flat 1.0 maxc)))
  (define divisor (where flat 1.0 span))
  (define (from c) (/ (- maxc c) divisor))
  (define rc (from r))
  (define gc (from g))
  (define bc (from b))
  (define h
    (where (eq maxc r)
           (- bc gc)
           (where (eq maxc g) (- (+ 2.0 rc) bc) (- (+ 4.0 gc) rc))))
  (values (~> (/ h 6.0) (+ 1.0) (fmod-scalar 1.0)) s maxc))

(define (hsv->rgb h s v)
  (define h6 (* h 6.0))
  (define whole (floor-tensor h6))
  (define f (- h6 whole))
  (define sector (fmod-scalar whole 6.0))
  (define p (unit (* v (- 1.0 s))))
  (define q (unit (* v (- 1.0 (* s f)))))
  (define t (unit (* v (- 1.0 (* s (- 1.0 f))))))
  (define (pick sector-0 . sectors)
    (for/fold ([acc sector-0])
              ([plane (in-list sectors)] [k (in-naturals 1)])
      (where (eq sector (exact->inexact k)) plane acc)))
  (cat (list (pick v q p p t v) (pick t v v q p p) (pick p p t v v q)) 1))

(define (turns t)
  (- t (floor-tensor t)))

(define (hue-turned x factor)
  (define-values (h s v) (rgb->hsv x))
  (~> (+ h factor) turns (hsv->rgb s v)))

(define (contrast-mean x)
  (mean-dim (luma x) '(1 2 3) #t #f))
