#lang racket/base

(module+ test
  (require (only-in rackunit check-equal? check-exn test-case)
           (only-in "../main.rkt" ones)
           (only-in "../nn.rkt" Linear Sequential define-layer forward
                    layer-forward))

  (define-layer Pair ()
    #:forward (x y)
    (list x y))

  (test-case "a layer called with the wrong number of inputs says which layer"
    (define seq (Sequential (Linear 2 2)))
    (define x (ones 1 2))
    (check-exn #rx"^Sequential: arity mismatch.*expected: 1.*given: 2"
               (lambda () (seq x x)))
    (check-exn exn:fail:contract:arity? (lambda () (seq x x)))
    (check-exn #rx"^Sequential: arity mismatch.*expected: 1.*given: 0"
               (lambda () (forward seq)))
    (check-exn #rx"^Sequential: arity mismatch.*given: 2"
               (lambda () (layer-forward seq x x)))
    (define p (Pair))
    (check-exn #rx"^Pair: arity mismatch.*expected: 2.*given: 1"
               (lambda () (p x)))
    (check-equal? (length (p x x)) 2)))
