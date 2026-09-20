#lang racket/base
(require torch torch/nn (only-in torch/foreign/contracts image-batch/c))

(define-layer Scale (w)
  #:init ()
  (set! w (Parameter (ones 2)))
  #:forward ([x : image-batch/c])
  (mul x w))

(define-layer Plain (w)
  #:init ()
  (set! w (Parameter (ones 2)))
  #:forward (x)
  (mul x w))

(define s (Scale))
(printf "ok rank 4: ~a\n" (tensor-shape (s (ones 1 1 1 2))))
(with-handlers ([exn:fail? (lambda (e) (printf "BAD RANK:\n~a\n" (exn-message e)))])
  (s (ones 2)))
(with-handlers ([exn:fail? (lambda (e) (printf "ARITY: ~a\n" (car (regexp-split #rx"\n" (exn-message e)))))])
  (s (ones 1 1 1 2) 3))
(printf "plain still works: ~a\n" (tensor-shape ((Plain) (ones 2))))
