#lang racket/base

(module+ test
  (require rackunit
           "../nn.rkt")

  (define-layer Twice (a b)
    #:init (child)
    (set! a child)
    (set! b child)
    #:forward (x)
    (b (a x)))

  (test-case "set-mode! sets the whole tree and answers the layer"
    (define child (Linear 2 2))
    (define m (Twice child))
    (check-eq? (set-mode! m 'eval) m)
    (check-false (layer-training? child))
    (check-eq? (set-mode! m 'train) m)
    (check-true (layer-training? child)))

  (test-case "call-with-eval-mode restores a child reachable twice"
    (define child (Linear 2 2))
    (define m (Twice child))
    (check-equal? (call-with-eval-mode m (lambda () (layer-training? child)))
                  #f)
    (check-true (layer-training? child))))
