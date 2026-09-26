#lang racket/base

(module+ test
  (require (only-in rackunit check-equal? check-exn check-true test-case)
           (only-in "../main.rkt"
                    arange copy! cuda-available? cuda-device in-flattened-tensor
                    in-tensor item reshape tensor tensor->list tensor-device
                    tensor-shape to zeros))

  (define m (reshape (arange 6) 2 3))

  (test-case "in-tensor yields the slices along the first dimension"
    (define rows (for/list ([row (in-tensor m)]) row))
    (check-equal? (map tensor-shape rows) '((3) (3)))
    (check-equal? (map tensor->list rows) '((0.0 1.0 2.0) (3.0 4.0 5.0)))
    (check-equal? (for/list ([plane (in-tensor (reshape (arange 24) 2 3 4))])
                    (tensor-shape plane))
                  '((3 4) (3 4))))

  (test-case "in-tensor over a vector yields zero-dimensional tensors"
    (define xs (for/list ([x (in-tensor (tensor '(7 8 9)))]) x))
    (check-equal? (map tensor-shape xs) '(() () ()))
    (check-equal? (map item xs) '(7 8 9)))

  (test-case "in-tensor's slices are views of the tensor"
    (define t (zeros 2 2))
    (for ([row (in-tensor t)] [v (in-naturals 1)])
      (copy! row (tensor (list v v) #:dtype 'float32)))
    (check-equal? (tensor->list t) '(1.0 1.0 2.0 2.0)))

  (test-case "in-tensor stops at the first dimension and needs one"
    (check-equal? (for/list ([row (in-tensor (zeros 0 3))]) row) '())
    (check-exn #rx"in-tensor: contract violation.*tensor-of-rank-at-least-one"
               (lambda () (in-tensor (tensor 5))))
    (check-exn #rx"in-tensor: contract violation"
               (lambda () (in-tensor '(1 2 3)))))

  (test-case "in-flattened-tensor yields the elements as numbers, row-major"
    (check-equal? (for/list ([x (in-flattened-tensor m)]) x)
                  '(0.0 1.0 2.0 3.0 4.0 5.0))
    (check-equal? (for/list ([i (in-flattened-tensor (tensor '((1 2) (3 4))))])
                    i)
                  '(1 2 3 4))
    (check-equal? (for/list ([x (in-flattened-tensor (tensor 5))]) x) '(5))
    (check-equal? (for/list ([x (in-flattened-tensor (zeros 0 3))]) x) '())
    (check-exn #rx"in-flattened-tensor: contract violation"
               (lambda () (in-flattened-tensor '(1 2)))))

  (test-case "the two walk in parallel, as topk's values and indices do"
    (check-equal? (for/list ([p (in-flattened-tensor (tensor '(0.5 0.25)))]
                             [i (in-flattened-tensor (tensor '(3 1)))])
                    (cons i p))
                  '((3 . 0.5) (1 . 0.25))))

  (test-case "on an accelerator the rows stay there"
    (when (cuda-available?)
      (define g (to m (cuda-device)))
      (check-true (for/and ([row (in-tensor g)])
                    (equal? (tensor-device row) (cuda-device))))
      (check-equal? (for/list ([x (in-flattened-tensor g)]) x)
                    '(0.0 1.0 2.0 3.0 4.0 5.0)))))
