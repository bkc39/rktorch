#lang racket/base

;; The nn layer: define-module expansion, parameter registration/recursion,
;; the linear layer, SGD, and a tiny end-to-end training loop. Seeded parity
;; with PyTorch's nn.Linear lives in python-cross-test.

(module+ test
  (require (except-in racket/list argmax flatten)
           rackunit
           "../main.rkt"
           "../nn.rkt")

  (define-module mlp (in hidden out)
    #:submodules ([fc1 (linear in hidden)]
                  [fc2 (linear hidden out)])
    #:forward (x)
    (fc2 (relu (fc1 x))))

  (define-module scale-shift (scale)
    #:buffers ([shift (ones 2)])
    #:forward (x)
    (add (mul x scale) shift))

  (test-case "linear layer: shapes, forward, predicate"
    (manual-seed! 0)
    (define l (linear 4 3))
    (check-true (linear? l))
    (check-true (module? l))
    (define ps (parameters l))
    (check-equal? (map tensor-shape ps) '((3 4) (3)))
    (check-true (andmap requires-grad? ps))
    (define y (l (randn 5 4)))
    (check-equal? (tensor-shape y) '(5 3))
    ;; (forward l x) and (l x) are the same entry point
    (manual-seed! 1)
    (define x (randn 2 4))
    (check-equal? (tensor->list (forward l x)) (tensor->list (l x))))

  (test-case "kaiming-uniform stays within the PyTorch bound"
    (manual-seed! 0)
    (define w (kaiming-uniform '(8 4)))
    ;; gain = sqrt(2/(1+5)) = sqrt(1/3); bound = sqrt(3)*gain/sqrt(4) = 0.5
    (for ([v (in-list (tensor->list w))])
      (check-true (and (>= v -0.5) (< v 0.5)))))

  (test-case "parameters recurse the module tree depth-first"
    (manual-seed! 0)
    (define net (mlp 4 8 2))
    (check-true (mlp? net))
    (define ps (parameters net))
    (check-equal? (map tensor-shape ps) '((8 4) (8) (2 8) (2)))
    (check-equal? (map car (named-parameters net))
                  '("fc1.weight" "fc1.bias" "fc2.weight" "fc2.bias"))
    (check-equal? (tensor-shape (net (randn 16 4))) '(16 2)))

  (test-case "ctor args and buffers are visible in forward; buffers not trained"
    (define m (scale-shift 3.0))
    (check-equal? (parameters m) '())
    (check-equal? (map tensor-shape (buffers m)) '((2)))
    (check-equal? (tensor->list (m (tensor '(1 2)))) '(4.0 7.0)))

  (test-case "sgd step applies p -= lr * grad and zero-grads! resets"
    (manual-seed! 0)
    (define l (linear 2 1))
    (define opt (sgd (parameters l) #:lr 0.5))
    (define x (tensor '((1.0 2.0))))
    (define y (tensor '((1.0))))
    (define before (map tensor->list (parameters l)))
    (define loss (mse-loss (l x) y))
    (backward! loss)
    (define grads (map (lambda (p) (tensor->list (grad p))) (parameters l)))
    (step! opt)
    (for ([p (in-list (parameters l))]
          [b (in-list before)]
          [g (in-list grads)])
      (for ([pv (in-list (tensor->list p))]
            [bv (in-list b)]
            [gv (in-list g)])
        (check-= pv (- bv (* 0.5 gv)) 1e-6)))
    (zero-grads! opt)
    (for ([p (in-list (parameters l))])
      (check-equal? (tensor->list (grad p))
                    (map (lambda (_) 0.0) (tensor->list p)))))

  (test-case "a few SGD steps reduce the training loss"
    (manual-seed! 0)
    (define net (mlp 4 8 2))
    (define opt (sgd (parameters net) #:lr 0.1))
    (define x (randn 16 4))
    (define y (randn 16 2))
    (define losses
      (for/list ([_ (in-range 5)])
        (zero-grads! opt)
        (define loss (mse-loss (net x) y))
        (backward! loss)
        (step! opt)
        (item loss)))
    (check-true (< (last losses) (first losses))
                (format "losses did not decrease: ~a" losses))))
