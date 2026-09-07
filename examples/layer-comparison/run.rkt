#lang racket/base
(require (only-in torch backward! mean mul randn tensor-shape with-no-grad)
         (only-in torch/nn adam eval! parameters step! train! zero-grads!)
         "models.rkt")

(define vision (SmallResNet #:classes 10))
(define transformer (TransformerStack 32 4 2 16 #:dropout 0.1))

(for ([net (in-list (list vision transformer))]
      [x (in-list (list (randn 2 3 32 32) (randn 2 8 32)))])
  (define optimizer (adam (parameters net) #:lr 1e-3))
  (train! net)
  (zero-grads! optimizer)
  (define output (net x))
  (define loss (mean (mul output output)))
  (backward! loss)
  (step! optimizer)
  (eval! net)
  (displayln (tensor-shape (with-no-grad (net x)))))
