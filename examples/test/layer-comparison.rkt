#lang racket/base

(require (only-in racket/list last)
         torch
         torch/nn
         "../layer-comparison/models.rkt")

(define (build)
  (manual-seed! 0)
  (values (SmallResNet) (TransformerStack 32 4 2 16)))

(module+ main
  (define-values (vision transformer) (build))
  (printf "SmallResNet ~a\n" (tensor-shape (vision (randn 2 3 32 32))))
  (printf "TransformerStack ~a\n" (tensor-shape (transformer (randn 2 8 32))))
  (for ([net (in-list (list vision transformer))])
    (printf "~a: ~a parameter tensors\n" (object-name net)
            (length (parameters net)))))

(module+ test
  (require rackunit)
  (define-values (vision transformer) (build))
  (check-equal? (tensor-shape (vision (randn 2 3 32 32))) '(2 10))
  (check-equal? (tensor-shape (transformer (randn 2 8 32))) '(2 8 32))
  (check-equal? (tensor-shape (transformer (randn 1 3 32))) '(1 3 32))
  (check-equal? (length (buffers transformer)) 2)
  (check-equal? (map car (named-buffers transformer))
                '("blocks.0.attention.mask" "blocks.1.attention.mask"))
  (define names (map car (named-parameters vision)))
  (check-equal? (car names) "stem.weight")
  (check-equal? (last names) "head.bias")
  (check-not-false (member "stage2.blocks.0.shortcut.weight" names))
  (check-false (member "stage1.blocks.0.shortcut.weight" names))
  (define x (randn 2 8 32))
  (backward! (mean (transformer x)))
  (check-true (andmap has-grad? (parameters transformer)))
  (define optimizer (adam (parameters transformer)))
  (step! optimizer)
  (void (eval! transformer))
  (check-equal? (tensor->list (transformer x)) (tensor->list (transformer x))))
