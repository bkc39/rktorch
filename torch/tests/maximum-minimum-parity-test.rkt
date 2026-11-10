#lang racket/base

(module+ test
  (require (only-in rackunit check-equal? check-exn check-true)
           (only-in "../main.rkt"
                    ~> maximum minimum reshape tensor tensor->list tensor-dtype
                    tensor-shape to-dtype)
           (only-in "private/blame.rkt" blames-caller?)
           "private/python-env.rkt")

  (define (read-value s)
    (read (open-input-string s)))

  (define (operand j)
    (define t
      (~> (map read-value (hash-ref j 'values))
          tensor
          (to-dtype (string->symbol (hash-ref j 'dtype)))))
    (apply reshape t (hash-ref j 'shape)))

  (define (same-value? r p)
    (cond
      [(and (flonum? r) (flonum? p) (not (= r r))) (not (= p p))]
      [else (equal? r p)]))

  (define (flat v)
    (if (list? v) (apply append (map flat v)) (list v)))

  (define (check-result label got want)
    (check-equal? (tensor-dtype got) (string->symbol (hash-ref want 'dtype))
                  (format "~a: dtype" label))
    (check-equal? (tensor-shape got) (hash-ref want 'shape)
                  (format "~a: shape" label))
    (define expected (map read-value (hash-ref want 'values)))
    (define answered (flat (tensor->list got)))
    (check-equal? (length answered) (length expected) (format "~a: numel" label))
    (for ([r (in-list answered)] [p (in-list expected)] [i (in-naturals)])
      (check-true (same-value? r p)
                  (format "~a: value ~a is ~a, torch says ~a" label i r p))))

  (define (check-case j)
    (define name (hash-ref j 'name))
    (define a (operand (hash-ref j 'a)))
    (define b (operand (hash-ref j 'b)))
    (check-result (format "maximum ~a" name) (maximum a b) (hash-ref j 'maximum))
    (check-result (format "minimum ~a" name) (minimum a b) (hash-ref j 'minimum)))

  (for ([extremum (in-list (list maximum minimum))])
    (check-exn (blames-caller? "maximum-minimum-parity-test")
               (lambda () (extremum (tensor '(1.0 2.0)) 1.5)))
    (check-exn (blames-caller? "maximum-minimum-parity-test")
               (lambda () (extremum 1.5 (tensor '(1.0 2.0))))))

  (cond
    [(not (python-torch-available?))
     (printf "[maximum-minimum-parity-test] skipped: python3 `torch` not available ~a\n"
             "(run inside the default `nix develop`)")]
    [else
     (for ([j (in-list (python-check "maximum_minimum.py"))])
       (check-case j))]))
