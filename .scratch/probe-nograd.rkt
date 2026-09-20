#lang racket/base
(require torch)

(define x (mul (rand 3 3 2 2 #:requires-grad? #t) 1.0))
(printf "x requires-grad? ~a\n" (requires-grad? x))

(define g (full 0.0 3 8 8))
(printf "fresh full requires-grad? ~a\n" (requires-grad? g))

(with-no-grad
  (printf "inside no-grad, grad-enabled? ~a\n" (grad-enabled?))
  (copy! (narrow (narrow g 1 0 2) 2 0 2) (select x 0 0))
  (printf "after copy! under no-grad, g requires-grad? ~a\n" (requires-grad? g)))

(define g2 (full 0.0 3 8 8))
(copy! (narrow (narrow g2 1 0 2) 2 0 2) (select x 0 0))
(printf "after copy! with grad on, g2 requires-grad? ~a\n" (requires-grad? g2))

(with-no-grad
  (printf "select under no-grad requires-grad? ~a\n"
          (requires-grad? (select x 0 0))))
