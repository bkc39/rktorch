#lang racket/base

(require (only-in racket/contract/base -> ->* listof)
         (only-in "../foreign.rkt" randn tensor? uniform! zeros)
         (only-in "../private/contract.rkt" define/checked-out))

(provide call-without-drawing
         uniform-bias)

(define dims/c (listof exact-nonnegative-integer?))

;; nn.TransformerEncoder's deep copies draw nothing, so a copy built through
;; its constructor must not draw either, or seeded parity drifts.
(define drawing? (make-parameter #t))

(define (call-without-drawing thunk)
  (parameterize ([drawing? #f])
    (thunk)))

;; zeros + uniform! consumes the RNG exactly as torch's empty().uniform_().
(define/checked-out (uniform-init dims low high)
  (-> dims/c real? real? tensor?)
  (define t (apply zeros dims))
  (when (drawing?)
    (uniform! t low high))
  t)

;; randn is empty().normal_(): the same RNG consumption as init.normal_.
(define/checked-out (normal-init dims) ;; noqa
  (-> dims/c tensor?)
  (if (drawing?) (apply randn dims) (apply zeros dims)))

(define/checked-out (fan-in dims)
  (-> dims/c exact-nonnegative-integer?)
  (apply * (cdr dims)))

;; the default #:a (sqrt 5) is what nn.Linear.reset_parameters passes
(define/checked-out (kaiming-uniform dims #:a [a (sqrt 5.0)]) ;; noqa
  (->* [dims/c] [#:a real?] tensor?)
  (define gain (sqrt (/ 2.0 (+ 1.0 (* a a)))))
  (define bound (* (sqrt 3.0) (/ gain (sqrt (fan-in dims)))))
  (uniform-init dims (- bound) bound))

;; the bias of nn.Linear and _ConvNd: U(-1/sqrt fan, 1/sqrt fan)
(define (uniform-bias n fan)
  (define bound (/ 1.0 (sqrt fan)))
  (uniform-init (list n) (- bound) bound))
