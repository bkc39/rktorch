#lang racket/base

(module+ test
  (require (only-in racket/list flatten)
           (only-in rackunit check-equal?)
           (only-in threading _ ~>)
           (only-in "../main.rkt"
                    maximum minimum reshape tensor tensor->list tensor-dtype
                    tensor-shape to-dtype)
           (only-in "private/python-env.rkt" python-check python-torch-available?))

  (define (operand j)
    (~> (map string->number (hash-ref j 'values))
        tensor
        (to-dtype (string->symbol (hash-ref j 'dtype)))
        (apply reshape _ (hash-ref j 'shape))))

  (define (check-result label got want)
    (check-equal? (tensor-dtype got) (string->symbol (hash-ref want 'dtype))
                  (format "~a: dtype" label))
    (check-equal? (tensor-shape got) (hash-ref want 'shape)
                  (format "~a: shape" label))
    (check-equal? (flatten (tensor->list got))
                  (map string->number (hash-ref want 'values))
                  (format "~a: values" label)))

  (define (check-case j)
    (define a (operand (hash-ref j 'a)))
    (define b (operand (hash-ref j 'b)))
    (for ([op (in-list (list maximum minimum))]
          [key (in-list '(maximum minimum))])
      (check-result (format "~a ~a" key (hash-ref j 'name)) (op a b) (hash-ref j key))))

  (cond
    [(not (python-torch-available?))
     (printf "[maximum-minimum-parity-test] skipped: python3 `torch` not available ~a\n"
             "(run inside the default `nix develop`)")]
    [else
     (for ([j (in-list (python-check "maximum_minimum.py"))])
       (check-case j))]))
