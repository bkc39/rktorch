#lang racket/base

;; The nn layer: define-module expansion, parameter registration/recursion,
;; the linear layer, SGD, and a tiny end-to-end training loop. Seeded parity
;; with PyTorch's nn.Linear lives in python-cross-test.

(module+ test
  ;; Layers are PascalCase (Conv2d/MaxPool2d/Flatten/Linear/…) since #11, so
  ;; `(require torch torch/nn)` no longer collides with the lowercase functional
  ;; ops on `torch`. (racket/list's flatten/argmax still collide with torch's
  ;; functional flatten/argmax, so those two are excepted.)
  (require (except-in racket/list argmax flatten)
           (only-in racket/file make-temporary-file)
           rackunit
           "../main.rkt"
           "../nn.rkt")

  (define-module mlp (in hidden out)
    #:submodules ([fc1 (Linear in hidden)]
                  [fc2 (Linear hidden out)])
    #:forward (x)
    (fc2 (relu (fc1 x))))

  (define-module scale-shift (scale)
    #:buffers ([shift (ones 2)])
    #:forward (x)
    (add (mul x scale) shift))

  (test-case "Linear layer: shapes, forward, predicate"
    (manual-seed! 0)
    (define l (Linear 4 3))
    (check-true (Linear? l))
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
    (define l (Linear 2 1))
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

  (test-case "Conv2d layer: param shapes, names, predicate, forward shape"
    (manual-seed! 0)
    (define c (Conv2d 1 8 3 #:stride 1 #:padding 1))
    (check-true (Conv2d? c))
    (check-true (module? c))
    (define ps (parameters c))
    (check-equal? (map tensor-shape ps) '((8 1 3 3) (8)))
    (check-true (andmap requires-grad? ps))
    (check-equal? (map car (named-parameters c)) '("weight" "bias"))
    (check-equal? (tensor-shape (c (randn 4 1 28 28))) '(4 8 28 28))
    ;; reflection-name tracks the public constructor, not the internal Conv2d%
    (check-equal? (object-name c) 'Conv2d))

  (test-case "Conv2d non-square kernel + per-axis padding"
    (manual-seed! 0)
    (define c (Conv2d 3 6 '(3 5) #:padding '(1 2)))
    (check-equal? (tensor-shape (car (parameters c))) '(6 3 3 5))
    (check-equal? (tensor-shape (c (randn 2 3 10 10))) '(2 6 10 10)))

  (test-case "MaxPool2d layer: stateless, default stride = kernel"
    (define p (MaxPool2d 2))
    (check-true (MaxPool2d? p))
    (check-equal? (parameters p) '())
    (check-equal? (tensor-shape (p (randn 4 8 28 28))) '(4 8 14 14))
    (check-equal? (object-name p) 'MaxPool2d))

  (test-case "Flatten layer: collapses from start-dim, keeps batch"
    (define f (Flatten))
    (check-true (Flatten? f))
    (check-equal? (parameters f) '())
    (check-equal? (tensor-shape (f (randn 4 8 14 14))) '(4 1568))
    (check-equal? (object-name f) 'Flatten))

  (test-case "conv -> pool -> flatten -> linear convnet composes"
    (manual-seed! 0)
    (define-module convnet ()
      #:submodules ([c1 (Conv2d 1 8 3 #:padding 1)]
                    [pool (MaxPool2d 2)]
                    [flat (Flatten)]
                    [fc (Linear (* 8 14 14) 10)])
      #:forward (x)
      (fc (flat (pool (relu (c1 x))))))
    (define net (convnet))
    (check-equal? (tensor-shape (net (randn 4 1 28 28))) '(4 10))
    ;; stateless submodules contribute no params; order is depth-first
    (check-equal? (map car (named-parameters net))
                  '("c1.weight" "c1.bias" "fc.weight" "fc.bias")))

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
                (format "losses did not decrease: ~a" losses)))

  (test-case "cross-entropy: known value, integer targets coerced to int64"
    (define logits (tensor '((-0.5 -1.0 -2.0) (-2.0 -0.2 -1.5))))
    (define targets (tensor '(0 1)))
    ;; = nll_loss(log_softmax(logits), targets), mean reduction
    (check-= (item (cross-entropy logits targets)) 0.48362 1e-4))

  (test-case "a few Adam steps reduce the training loss"
    (manual-seed! 0)
    (define net (mlp 4 8 2))
    (define opt (adam (parameters net) #:lr 0.05))
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
                (format "Adam losses did not decrease: ~a" losses)))

  (test-case "dropout: train drops/scales, eval is identity, mode recurses"
    (manual-seed! 0)
    (define d (Dropout #:p 0.5))
    (define x (ones 100))
    ;; training (default): each entry is 0 or 2.0 (kept and scaled by 1/(1-p))
    (define tr (tensor->list (d x)))
    (check-true (andmap (lambda (v) (or (= v 0.0) (= v 2.0))) tr))
    (check-true (> (length (filter zero? tr)) 0) "nothing was dropped")
    ;; eval: identity
    (eval! d)
    (check-equal? (tensor->list (d x)) (tensor->list x))
    ;; train! flips back
    (train! d)
    (check-true (andmap (lambda (v) (or (= v 0.0) (= v 2.0)))
                        (tensor->list (d x)))))

  (test-case "dropout inside a model: eval! recurses through submodules"
    (define net (Sequential (Linear 4 4) (Dropout #:p 0.9)))
    (eval! net)
    ;; with dropout off, repeated forwards on the same input agree
    (define x (randn 2 4))
    (check-equal? (tensor->list (net x)) (tensor->list (net x))))

  (test-case "module-training? + in-eval-mode: query and restore the prior mode"
    ;; a mode-sensitive leaf reports + toggles its own flag
    (define d (Dropout #:p 0.5))
    (check-true (module-training? d) "dropout defaults to training")
    (in-eval-mode d (check-false (module-training? d) "eval inside the body"))
    (check-true (module-training? d) "restored to train")
    ;; restores to the *prior* mode, not unconditionally train: from eval -> eval
    (eval! d)
    (in-eval-mode d (check-false (module-training? d)))
    (check-false (module-training? d) "restored to eval, not flipped to train")
    ;; structural: module-training? recurses (define-module linear leaves are
    ;; always #t; the dropout child carries the mode) and in-eval-mode restores
    (train! d)
    (define net (Sequential (Linear 4 4) (Dropout #:p 0.5)))
    (check-true (module-training? net))
    (in-eval-mode net (check-false (module-training? net)))
    (check-true (module-training? net) "model restored to train")
    ;; a dropout-free define-module is vacuously training (eval! a no-op)
    (define lin (Linear 4 2))
    (check-true (module-training? lin))
    (in-eval-mode lin (check-true (module-training? lin)))
    (check-true (module-training? lin)))

  (test-case "sequential: forward, indexed dotted names, param order"
    (manual-seed! 0)
    (define net (Sequential (Linear 4 8) (Dropout #:p 0.5) (Linear 8 2)))
    (check-equal? (tensor-shape (net (randn 3 4))) '(3 2))
    (check-equal? (map car (named-parameters net))
                  '("0.weight" "0.bias" "2.weight" "2.bias"))
    (check-equal? (length (parameters net)) 4)
    (check-true (Sequential? net)))

  (test-case "safetensors state-dict round-trips bit-exactly"
    (manual-seed! 0)
    (define net (Sequential (Linear 4 8) (Dropout #:p 0.3) (Linear 8 2)))
    (define path (make-temporary-file "rkt-st-~a.safetensors"))
    (save-state! net path)
    ;; a differently-seeded model has different params...
    (manual-seed! 99)
    (define net2 (Sequential (Linear 4 8) (Dropout #:p 0.3) (Linear 8 2)))
    (load-state! net2 path)
    ;; ...and after load matches the saved model parameter-for-parameter.
    (for ([a (in-list (state-dict net))] [b (in-list (state-dict net2))])
      (check-equal? (car a) (car b))
      (check-equal? (tensor->list (cdr a)) (tensor->list (cdr b))))
    (delete-file path)))
