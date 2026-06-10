#lang racket/base

;; Parameter initializers, built purely from tensor ops — no new C++. Each
;; allocates the tensor and fills it with uniform! (ATen's uniform_), so the
;; global-RNG consumption matches torch.nn.init exactly; that is what makes
;; seeded layer init bit-comparable with PyTorch.

;; Numeric math here stays racket/base (no tensors in the bounds), so only
;; the tensor constructors come from the facade.
(require (only-in "../foreign.rkt" uniform! zeros))

(provide uniform-init
         kaiming-uniform
         fan-in)

;; A dims-shaped tensor of uniform draws on [low, high). PyTorch fills
;; torch.empty via uniform_; zeros + uniform! consumes the RNG identically.
(define (uniform-init dims low high)
  (define t (apply zeros dims))
  (uniform! t low high)
  t)

;; torch.nn.init._calculate_fan_in_and_fan_out's fan_in: for a [out in ...]
;; weight, the input fmaps times the receptive field.
(define (fan-in dims)
  (apply * (cdr dims)))

;; torch.nn.init.kaiming_uniform_ (fan-in mode, leaky_relu nonlinearity).
;; The default #:a (sqrt 5) is what nn.Linear.reset_parameters passes.
(define (kaiming-uniform dims #:a [a (sqrt 5.0)])
  (define gain (sqrt (/ 2.0 (+ 1.0 (* a a)))))
  (define bound (* (sqrt 3.0) (/ gain (sqrt (fan-in dims)))))
  (uniform-init dims (- bound) bound))
