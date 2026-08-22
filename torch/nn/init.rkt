#lang racket/base

;; Parameter initializers. Each fills its tensor through ATen's own RNG ops,
;; so the global-RNG consumption matches torch.nn.init draw for draw — that
;; is what makes seeded layer init bit-comparable with PyTorch.

(require (only-in "../foreign.rkt" randn uniform! zeros))

(provide uniform-init
         normal-init
         kaiming-uniform
         fan-in)

;; PyTorch fills torch.empty via uniform_; zeros + uniform! consumes the RNG
;; identically.
(define (uniform-init dims low high)
  (define t (apply zeros dims))
  (uniform! t low high)
  t)

;; torch.randn is empty().normal_(), so the RNG consumption matches
;; torch.nn.init.normal_ draw for draw.
(define (normal-init dims)
  (apply randn dims))

;; torch.nn.init._calculate_fan_in_and_fan_out's fan_in for a [out in ...]
;; weight: input fmaps times receptive field.
(define (fan-in dims)
  (apply * (cdr dims)))

;; torch.nn.init.kaiming_uniform_ (fan-in mode, leaky_relu nonlinearity).
;; The default #:a (sqrt 5) is what nn.Linear.reset_parameters passes.
(define (kaiming-uniform dims #:a [a (sqrt 5.0)])
  (define gain (sqrt (/ 2.0 (+ 1.0 (* a a)))))
  (define bound (* (sqrt 3.0) (/ gain (sqrt (fan-in dims)))))
  (uniform-init dims (- bound) bound))
