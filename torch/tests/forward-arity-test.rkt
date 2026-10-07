#lang racket/base

(module+ test
  (require (only-in racket/contract/base flat-named-contract)
           (only-in rackunit check-equal? check-exn test-case)
           (only-in "../main.rkt" ones tensor-shape)
           (only-in "../nn.rkt"
                    Linear Sequential define-layer gen:layer with-mode))

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
               (lambda () (seq)))
    (define p (Pair))
    (check-exn #rx"^Pair: arity mismatch.*expected: 2.*given: 1"
               (lambda () (p x)))
    (check-equal? (length (p x x)) 2))

  (define-layer Scaled (factor)
    #:forward ([x : rank2/c] #:by [by factor] #:shift shift)
    (with-mode (list x by shift mode)))

  (define-layer Tagged ()
    #:forward (x #:tag [tag 'none] . more)
    (list x tag more))

  (struct Hand ()
    #:methods gen:layer
    [(define (layer-forward self . inputs) inputs)])

  (test-case "keyword inputs, with defaults that see the fields"
    (define s (Scaled 2))
    (define x (ones 1 2))
    (check-equal? (cdr ((Scaled 2) x #:shift 0)) '(2 0 train))
    (check-equal? (cdr (s x #:by 3 #:shift 1)) '(3 1 train))
    (check-equal? ((Tagged) 1) '(1 none ()))
    (check-equal? ((Tagged) 1 2 3 #:tag 'kw) '(1 kw (2 3)))
    (check-exn #rx"required keyword argument not supplied.*#:shift"
               (lambda () (s x)))
    (check-exn #rx"procedure: Scaled.*given keyword: #:nope"
               (lambda () (s x #:shift 0 #:nope 1)))
    (check-exn #rx"^Scaled: arity mismatch.*expected: 1.*given: 2"
               (lambda () (s x x #:shift 0)))
    (check-exn #rx"^Scaled: contract violation.*expected: rank2"
               (lambda () (s (ones 2) #:shift 0))))

  (test-case "a forward without keyword inputs refuses keywords by name"
    (check-exn #rx"does not accept keyword arguments.*procedure: Sequential"
               (lambda () ((Sequential (Linear 2 2)) (ones 1 2) #:foo 1)))
    (check-exn #rx"does not accept keyword arguments.*procedure: layer-forward"
               (lambda () ((Hand) 1 #:foo 1)))
    (check-equal? ((Hand) 1 2) '(1 2)))

  (test-case "a rest formal and a contracted one compose"
    (define s (Stateful))
    (define x (ones 1 2))
    (check-equal? (length (s x)) 1 "the rest may be empty")
    (check-equal? (length (s x 'h 'c)) 3)
    (check-exn #rx"^Stateful: contract violation"
               (lambda () (s (ones 1 2 3) 'h)))
    (check-exn #rx"expected: rank2" (lambda () (s (ones 1 2 3))))
    (check-exn #rx"^Stateful: arity mismatch" (lambda () (s)))))
