#lang racket/base

(require (only-in racket/contract/base -> ->* listof)
         (only-in "../foreign.rkt" randn tensor? uniform! zeros)
         (only-in "../private/contract.rkt" define/checked-out))

(define dims/c (listof exact-nonnegative-integer?))

;; zeros + uniform! consumes the RNG exactly as torch's empty().uniform_().
(define/checked-out (uniform-init dims low high)
  (-> dims/c real? real? tensor?)
  (define t (apply zeros dims))
  (uniform! t low high)
  t)

;; randn is empty().normal_(): the same RNG consumption as init.normal_.
(define/checked-out (normal-init dims) ;; noqa
  (-> dims/c tensor?)
  (apply randn dims))

(define/checked-out (fan-in dims)
  (-> dims/c exact-nonnegative-integer?)
  (apply * (cdr dims)))

;; the default #:a (sqrt 5) is what nn.Linear.reset_parameters passes
(define/checked-out (kaiming-uniform dims #:a [a (sqrt 5.0)]) ;; noqa
  (->* [dims/c] [#:a real?] tensor?)
  (define gain (sqrt (/ 2.0 (+ 1.0 (* a a)))))
  (define bound (* (sqrt 3.0) (/ gain (sqrt (fan-in dims)))))
  (uniform-init dims (- bound) bound))
