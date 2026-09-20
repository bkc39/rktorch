#lang racket/base

(module+ test
  (require (only-in racket/contract/base flat-named-contract)
           (only-in rackunit check-equal? check-exn test-case)
           (only-in "../main.rkt" ones tensor-shape)
           (only-in "../nn.rkt" Linear Sequential define-layer forward
                    layer-forward))

  (define-layer Pair ()
    #:forward (x y)
    (list x y))

  (define rank2/c
    (flat-named-contract 'rank2 (lambda (t) (= 2 (length (tensor-shape t))))))

  (define-layer Stateful ()
    #:forward ([x : rank2/c] . state)
    (cons x state))

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
    (check-equal? (length (p x x)) 2))

  (test-case "a rest formal and a contracted one compose"
    (define s (Stateful))
    (define x (ones 1 2))
    (check-equal? (length (s x)) 1 "the rest may be empty")
    (check-equal? (length (s x 'h 'c)) 3)
    (check-exn #rx"^Stateful: contract violation"
               (lambda () (s (ones 1 2 3) 'h)))
    (check-exn #rx"expected: rank2" (lambda () (s (ones 1 2 3))))
    (check-exn #rx"^Stateful: arity mismatch" (lambda () (s)))))
