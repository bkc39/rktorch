#lang racket/base

(module+ test
  (require rackunit
           "../main.rkt"
           "../nn.rkt")

  (define-layer Counted (w runs)
    #:init (counter #:weight [weight (Parameter (tensor '(1.0 2.0)))])
    (set! w weight)
    (set! runs counter)
    #:on-move (set-box! runs (add1 (unbox runs)))
    #:forward (x)
    (* x w))

  (define-layer Pair (a b)
    #:init (left right)
    (set! a left)
    (set! b right)
    #:forward (x)
    (b (a x)))

  (define-layer Steps (steps)
    #:init ()
    (set! steps (Buffer (tensor '(0 1))))
    #:on-move (error 'Steps "an integer buffer does not move under a dtype")
    #:forward (x)
    x)

  (test-case "the body runs only for a move that rebinds a tensor"
    (define runs (box 0))
    (define m (Counted runs))
    (to m 'cpu)
    (to m 'float32)
    (check-equal? (unbox runs) 0 "an identity move ran the body")
    (to m 'float64)
    (check-equal? (unbox runs) 1)
    (to m 'float32)
    (check-equal? (unbox runs) 2 "a round trip runs it once per move")
    (check-pred layer? (to (Steps) 'float64)))

  (test-case "moving a parent runs its children's bodies"
    (define left (box 0))
    (define right (box 0))
    (define m (Pair (Counted left) (Counted right)))
    (to m 'cpu)
    (check-equal? (list (unbox left) (unbox right)) '(0 0))
    (to m 'float64)
    (check-equal? (list (unbox left) (unbox right)) '(1 1)))

  (test-case "a tensor tied across siblings counts as moved for both"
    (define shared (Parameter (tensor '(1.0 2.0))))
    (define left (box 0))
    (define right (box 0))
    (define m (Pair (Counted left #:weight shared)
                    (Counted right #:weight shared)))
    (to m 'float64)
    (check-equal? (unbox left) 1)
    (check-equal? (unbox right) 1
                  "the second sibling found its tensor already moved"))

  (test-case "a child reachable twice runs its body once"
    (define runs (box 0))
    (define child (Counted runs))
    (define m (Pair child child))
    (to m 'float64)
    (check-equal? (unbox runs) 1)
    (check-equal? (tensor-dtype (car (parameters m))) 'float64)))
