#lang racket/base

(require (only-in "../foreign.rkt" randn uniform! zeros))

(provide uniform-init
         normal-init
         kaiming-uniform
         fan-in)

;; zeros + uniform! consumes the RNG exactly as torch's empty().uniform_().
(define (uniform-init dims low high)
  (define t (apply zeros dims))
  (uniform! t low high)
  t)

;; randn is empty().normal_(): the same RNG consumption as init.normal_.
(define (normal-init dims)
  (apply randn dims))

(define (fan-in dims)
  (apply * (cdr dims)))

;; the default #:a (sqrt 5) is what nn.Linear.reset_parameters passes
(define (kaiming-uniform dims #:a [a (sqrt 5.0)])
  (define gain (sqrt (/ 2.0 (+ 1.0 (* a a)))))
  (define bound (* (sqrt 3.0) (/ gain (sqrt (fan-in dims)))))
  (uniform-init dims (- bound) bound))
