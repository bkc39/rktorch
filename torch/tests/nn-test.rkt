#lang racket/base

(module+ test
  (require (except-in racket/list argmax flatten take)
           (only-in racket/file make-temporary-file)
           rackunit
           "../main.rkt"
           "../nn.rkt")

  (define-layer mlp (fc1 fc2)
    #:init (in hidden out)
    (set! fc1 (Linear in hidden))
    (set! fc2 (Linear hidden out))
    #:forward (x)
    (fc2 (relu (fc1 x))))

  (define-layer scale-shift (scale shift)
    #:init (scale)
    (set! shift (Buffer (ones 2)))
    #:forward (x)
    (add (mul x scale) shift))

  (test-case "Linear layer: shapes, forward, predicate"
    (manual-seed! 0)
    (define l (Linear 4 3))
    (check-true (linear? l))
    (check-true (layer? l))
    (check-equal? (object-name l) 'Linear)
    (define ps (parameters l))
    (check-equal? (map tensor-shape ps) '((3 4) (3)))
    (check-true (andmap requires-grad? ps))
    (define y (l (randn 5 4)))
    (check-equal? (tensor-shape y) '(5 3))
    (manual-seed! 1)
    (define x (randn 2 4))
    (check-equal? (tensor->list (forward l x)) (tensor->list (l x))))

  (test-case "kaiming-uniform stays within the PyTorch bound"
    (manual-seed! 0)
    (define w (kaiming-uniform '(8 4)))
    ;; bound = sqrt(3) * sqrt(2/(1+5)) / sqrt(fan-in 4) = 0.5
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
    (check-true (conv2d? c))
    (check-true (layer? c))
    (define ps (parameters c))
    (check-equal? (map tensor-shape ps) '((8 1 3 3) (8)))
    (check-true (andmap requires-grad? ps))
    (check-equal? (map car (named-parameters c)) '("weight" "bias"))
    (check-equal? (tensor-shape (c (randn 4 1 28 28))) '(4 8 28 28))
    (check-equal? (object-name c) 'Conv2d))

  (test-case "Conv2d non-square kernel + per-axis padding"
    (manual-seed! 0)
    (define c (Conv2d 3 6 '(3 5) #:padding '(1 2)))
    (check-equal? (tensor-shape (car (parameters c))) '(6 3 3 5))
    (check-equal? (tensor-shape (c (randn 2 3 10 10))) '(2 6 10 10)))

  (test-case "Conv1d layer: param shapes, names, predicate, forward shape"
    (manual-seed! 0)
    (define c (Conv1d 2 8 3 #:stride 1 #:padding 1))
    (check-true (conv1d? c))
    (check-true (layer? c))
    (define ps (parameters c))
    (check-equal? (map tensor-shape ps) '((8 2 3) (8)))
    (check-true (andmap requires-grad? ps))
    (check-equal? (map car (named-parameters c)) '("weight" "bias"))
    (check-equal? (tensor-shape (c (randn 4 2 100))) '(4 8 100))
    (check-equal? (object-name c) 'Conv1d)
    (define dilated (Conv1d 2 8 3 #:dilation 4 #:padding 4))
    (check-equal? (tensor-shape (dilated (randn 4 2 100))) '(4 8 100))
    (check-equal? (tensor->list
                   (conv1d (tensor '(((1.0 2.0 3.0 4.0 5.0))))
                           (ones 1 1 2)
                           #:dilation 2))
                  '(4.0 6.0 8.0)))

  (test-case "MaxPool2d layer: stateless, default stride = kernel"
    (define p (MaxPool2d 2))
    (check-true (max-pool2d? p))
    (check-equal? (parameters p) '())
    (check-equal? (tensor-shape (p (randn 4 8 28 28))) '(4 8 14 14))
    (check-equal? (object-name p) 'MaxPool2d))

  (test-case "Flatten layer: collapses from start-dim, keeps batch"
    (define f (Flatten))
    (check-true (flatten? f))
    (check-equal? (parameters f) '())
    (check-equal? (tensor-shape (f (randn 4 8 14 14))) '(4 1568))
    (check-equal? (object-name f) 'Flatten))

  (test-case "Embedding layer: weight shape, gather forward, predicate"
    (manual-seed! 0)
    (define e (Embedding 7 4))
    (check-true (embedding? e))
    (check-true (layer? e))
    (define ps (parameters e))
    (check-equal? (map tensor-shape ps) '((7 4)))
    (check-true (andmap requires-grad? ps))
    (check-equal? (map car (named-parameters e)) '("weight"))
    (define idx (to-dtype (tensor '(3 0 3)) 'int64))
    (define out (e idx))
    (check-equal? (tensor-shape out) '(3 4))
    (define w-rows (tensor->list (car ps)))
    (check-equal? (take (tensor->list out) 4)          ; row 3
                  (take (drop w-rows 12) 4))
    (check-equal? (object-name e) 'Embedding))

  (test-case "LayerNorm layer: ones/zeros init, normalizing forward"
    (define ln (LayerNorm 4))
    (check-true (layer-norm? ln))
    (check-true (layer? ln))
    (check-equal? (map car (named-parameters ln)) '("weight" "bias"))
    (check-equal? (map tensor-shape (parameters ln)) '((4) (4)))
    (check-equal? (tensor->list (car (parameters ln))) '(1.0 1.0 1.0 1.0))
    (check-equal? (tensor->list (cadr (parameters ln))) '(0.0 0.0 0.0 0.0))
    (define out (ln (tensor '((1.0 2.0 3.0 4.0) (10.0 20.0 30.0 40.0)))))
    (check-equal? (tensor-shape out) '(2 4))
    (define rows (tensor->list out))
    (check-= (apply + (take rows 4)) 0.0 1e-4)
    (check-= (apply + (drop rows 4)) 0.0 1e-4)
    (check-true (layer-norm? (LayerNorm '(3 4) #:eps 1e-6)))
    (check-equal? (object-name ln) 'LayerNorm))

  (test-case "#:reflection-name may precede other clauses (any-order)"
    (define-layer early-refl% (w)
      #:reflection-name 'EarlyRefl
      #:init ()
      (set! w (Parameter (zeros 2 2)))
      #:forward (x) (matmul x w))
    (check-equal? (object-name (early-refl%)) 'EarlyRefl))

  (test-case "conv -> pool -> flatten -> linear convnet composes"
    (manual-seed! 0)
    (define-layer convnet (c1 pool flat fc)
      #:init ()
      (set! c1 (Conv2d 1 8 3 #:padding 1))
      (set! pool (MaxPool2d 2))
      (set! flat (Flatten))
      (set! fc (Linear (* 8 14 14) 10))
      #:forward (x)
      (fc (flat (pool (relu (c1 x))))))
    (define net (convnet))
    (check-equal? (tensor-shape (net (randn 4 1 28 28))) '(4 10))
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
    (check-= (item (cross-entropy logits targets)) 0.48362 1e-4))

  (test-case "ctc-loss: closed form on uniform log-probs"
    ;; two frames, two classes, label 1: the alignments [1 1], [0 1]
    ;; and [1 0] carry probability 3/4, so the loss is -ln(3/4)
    (define log-half (log 0.5))
    (define log-probs
      (tensor (list (list (list log-half log-half))
                    (list (list log-half log-half)))))
    (define targets (tensor '((1))))
    (check-= (item (ctc-loss log-probs targets
                             #:input-lengths '(2)
                             #:target-lengths '(1)))
             0.2876821 1e-6)
    (check-exn #rx"input-lengths"
               (lambda () (ctc-loss log-probs targets
                                    #:input-lengths '()
                                    #:target-lengths '(1))))
    ;; a 0 target length is a valid empty transcript: only the all-blank
    ;; path survives, p = 1/4, and mean reduction clamps the divisor to 1
    (check-= (item (ctc-loss log-probs targets
                             #:input-lengths '(2)
                             #:target-lengths '(0)))
             1.3862944 1e-6)
    (check-exn #rx"blank"
               (lambda () (ctc-loss log-probs targets
                                    #:input-lengths '(2)
                                    #:target-lengths '(1)
                                    #:blank -1)))
    (check-exn #rx"stride"
               (lambda () (conv1d (randn 1 2 8) (randn 3 2 3) #:stride 0)))
    (check-exn #rx"padding"
               (lambda () (conv1d (randn 1 2 8) (randn 3 2 3)
                                  #:padding -1))))

  (test-case "ctc-loss on mps: same value, gradient back on the device"
    ;; libtorch has no MPS ctc_loss kernel, so the loss detours through the
    ;; CPU; the detour must be invisible in both the value and the gradient
    (when (mps-available?)
      (manual-seed! 0)
      (define frames (tensor->list (randn 6 2 5)))
      (define targets '((1 2 3) (2 3 1)))
      (define (loss+grad dev)
        (define w (to-device (reshape (tensor frames) 6 2 5) dev))
        (requires-grad! w)
        (define loss
          (ctc-loss (log-softmax w 2)
                    (to-device (to-dtype (tensor targets) 'int64) dev)
                    #:input-lengths '(6 5)
                    #:target-lengths '(3 3)
                    #:blank 4
                    #:zero-infinity? #t))
        (backward! loss)
        (values loss (grad w)))
      (define-values (cpu-loss cpu-grad) (loss+grad 'cpu))
      (define-values (mps-loss mps-grad) (loss+grad 'mps))
      (check-equal? (tensor-device mps-loss) (mps-device))
      (check-equal? (tensor-device mps-grad) (mps-device))
      (check-= (item mps-loss) (item cpu-loss) 1e-5)
      (for ([c (in-list (tensor->list cpu-grad))]
            [m (in-list (tensor->list (to-device mps-grad 'cpu)))])
        (check-= m c 1e-5))))

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
    (check-equal? (object-name d) 'Dropout)
    (define x (ones 100))
    (define tr (tensor->list (d x)))
    (check-true (andmap (lambda (v) (or (= v 0.0) (= v 2.0))) tr))
    (check-true (> (length (filter zero? tr)) 0) "nothing was dropped")
    (eval! d)
    (check-equal? (tensor->list (d x)) (tensor->list x))
    (train! d)
    (check-true (andmap (lambda (v) (or (= v 0.0) (= v 2.0)))
                        (tensor->list (d x)))))

  (test-case "dropout inside a model: eval! recurses through submodules"
    (define net (Sequential (Linear 4 4) (Dropout #:p 0.9)))
    (eval! net)
    (define x (randn 2 4))
    (check-equal? (tensor->list (net x)) (tensor->list (net x))))

  (test-case "layer-training? + in-eval-mode: query and restore the prior mode"
    (define d (Dropout #:p 0.5))
    (check-true (layer-training? d) "dropout defaults to training")
    (in-eval-mode d (check-false (layer-training? d) "eval inside the body"))
    (check-true (layer-training? d) "restored to train")
    ;; restores to the *prior* mode, not unconditionally train: from eval -> eval
    (eval! d)
    (in-eval-mode d (check-false (layer-training? d)))
    (check-false (layer-training? d) "restored to eval, not flipped to train")
    (train! d)
    (define net (Sequential (Linear 4 4) (Dropout #:p 0.5)))
    (check-true (layer-training? net))
    (in-eval-mode net (check-false (layer-training? net)))
    (check-true (layer-training? net) "model restored to train")
    (define lin (Linear 4 2))
    (check-true (layer-training? lin))
    (in-eval-mode lin (check-true (layer-training? lin)))
    (check-true (layer-training? lin)))

  (test-case "sequential: forward, indexed dotted names, param order"
    (manual-seed! 0)
    (define net (Sequential (Linear 4 8) (Dropout #:p 0.5) (Linear 8 2)))
    (check-equal? (tensor-shape (net (randn 3 4))) '(3 2))
    (check-equal? (map car (named-parameters net))
                  '("0.weight" "0.bias" "2.weight" "2.bias"))
    (check-equal? (length (parameters net)) 4)
    (check-true (sequential? net))
    (check-equal? (object-name net) 'Sequential))

  (test-case "safetensors state-dict round-trips bit-exactly"
    (manual-seed! 0)
    (define net (Sequential (Linear 4 8) (Dropout #:p 0.3) (Linear 8 2)))
    (define path (make-temporary-file "rkt-st-~a.safetensors"))
    (save-state! net path)
    ;; seed 99: net2 starts different, so post-load equality is non-vacuous
    (manual-seed! 99)
    (define net2 (Sequential (Linear 4 8) (Dropout #:p 0.3) (Linear 8 2)))
    (load-state! net2 path)
    (for ([a (in-list (state-dict net))] [b (in-list (state-dict net2))])
      (check-equal? (car a) (car b))
      (check-equal? (tensor->list (cdr a)) (tensor->list (cdr b))))
    (delete-file path)))
