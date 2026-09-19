#lang racket/base

(module+ test
  (require (except-in racket/list argmax flatten take)
           (only-in racket/file file->bytes make-temporary-file)
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

  (test-case "sgd momentum, Nesterov and weight decay follow torch.optim.SGD"
    ;; loss = sum(w): the gradient is 1 everywhere, so the update is the
    ;; buffer arithmetic alone
    (define (trained #:momentum [mu 0.0] #:nesterov? [nesterov? #f]
                     #:weight-decay [wd 0.0] #:steps [steps 2])
      (define w (Parameter (ones 2)))
      (define opt (sgd (list w) #:lr 0.1 #:momentum mu #:nesterov? nesterov?
                       #:weight-decay wd))
      (for ([_ (in-range steps)])
        (zero-grads! opt)
        (backward! (sum w))
        (step! opt))
      (car (tensor->list w)))
    (check-= (trained) 0.8 1e-6 "plain: two steps of lr")
    ;; buffers 1 then 1.9: 1 - 0.1 - 0.19
    (check-= (trained #:momentum 0.9) 0.71 1e-6)
    ;; Nesterov looks ahead: 1 + 0.9 * 1 = 1.9 on the first step already
    (check-= (trained #:momentum 0.9 #:nesterov? #t #:steps 1) 0.81 1e-6)
    ;; decay adds wd * w to the gradient: 1 + 0.5 * 1 = 1.5 on the first step
    (check-= (trained #:weight-decay 0.5 #:steps 1) 0.85 1e-6)
    (check-exn #rx"Nesterov momentum requires a momentum"
               (lambda () (sgd (list (Parameter (ones 1))) #:lr 0.1
                               #:nesterov? #t))))

  (test-case "rmsprop divides by the root of the running square average"
    (define w (Parameter (ones 2)))
    (define opt (rmsprop (list w) #:lr 0.01))
    (check-true (rmsprop? opt))
    (zero-grads! opt)
    (backward! (sum w))
    (step! opt)
    ;; v = 0.01, sqrt v = 0.1: the first step moves by lr * 1 / 0.1
    (check-= (car (tensor->list w)) 0.9 1e-5)
    (define m (Parameter (ones 2)))
    (define with-momentum (rmsprop (list m) #:lr 0.01 #:momentum 0.5))
    (zero-grads! with-momentum)
    (backward! (sum m))
    (step! with-momentum)
    (check-= (car (tensor->list m)) 0.9 1e-5 "the buffer starts at zero")
    (check-= (learning-rate opt) 0.01 0.0)
    (set-learning-rate! opt 0.02)
    (check-= (learning-rate opt) 0.02 0.0))

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

  (test-case "Conv2d without a bias draws the weight alone"
    (manual-seed! 0)
    (define with-bias (Conv2d 1 8 3))
    (manual-seed! 0)
    (define bare (Conv2d 1 8 3 #:bias? #f))
    (check-equal? (map tensor-shape (parameters bare)) '((8 1 3 3)))
    (check-equal? (map car (named-parameters bare)) '("weight"))
    (check-equal? (tensor->list (car (parameters bare)))
                  (tensor->list (car (parameters with-bias)))
                  "the same weight draw")
    (check-equal? (tensor-shape (bare (randn 2 1 8 8))) '(2 8 6 6)))

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

  (test-case "ConvTranspose2d layer: transposed weight layout, forward shape"
    (manual-seed! 0)
    (define c (ConvTranspose2d 3 6 4 #:stride 2 #:padding 1))
    (check-true (conv-transpose2d? c))
    (check-true (layer? c))
    (define ps (parameters c))
    (check-equal? (map tensor-shape ps) '((3 6 4 4) (6)))
    (check-true (andmap requires-grad? ps))
    (check-equal? (map car (named-parameters c)) '("weight" "bias"))
    (check-equal? (tensor-shape (c (randn 2 3 8 8))) '(2 6 16 16))
    (check-equal? (object-name c) 'ConvTranspose2d)
    (check-equal? (map tensor-shape (parameters (ConvTranspose2d 4 4 3 #:groups 2)))
                  '((4 2 3 3) (4)))
    (check-equal? (tensor-shape ((ConvTranspose2d 4 6 3 #:groups 2 #:dilation 2)
                                 (randn 1 4 4 4)))
                  '(1 6 8 8))
    (check-equal? (tensor-shape ((ConvTranspose2d 1 1 2 #:stride 2 #:output-padding 1)
                                 (ones 1 1 2 2)))
                  '(1 1 5 5)))

  (test-case "GroupNorm layer: ones/zeros init, per-group normalizing forward"
    (define gn (GroupNorm 2 4))
    (check-true (group-norm? gn))
    (check-equal? (map tensor-shape (parameters gn)) '((4) (4)))
    (check-equal? (map car (named-parameters gn)) '("weight" "bias"))
    (check-equal? (tensor->list (car (parameters gn))) '(1.0 1.0 1.0 1.0))
    (check-equal? (tensor->list (cadr (parameters gn))) '(0.0 0.0 0.0 0.0))
    (manual-seed! 0)
    (define y (gn (randn 2 4 3 3)))
    (check-equal? (tensor-shape y) '(2 4 3 3))
    (define grouped (reshape y 2 2 18))
    (for* ([b (in-range 2)] [g (in-range 2)])
      (check-= (item (mean (select (select grouped 0 b) 0 g))) 0.0 1e-5))
    (check-equal? (object-name gn) 'GroupNorm))

  (test-case "BatchNorm2d layer: init, batch statistics, running statistics, eval"
    (define bn (BatchNorm2d 3))
    (check-true (batch-norm2d? bn))
    (check-equal? (map car (named-parameters bn)) '("weight" "bias"))
    (check-equal? (map car (named-buffers bn))
                  '("running-mean" "running-var" "num-batches-tracked"))
    (check-equal? (map tensor-shape (buffers bn)) '((3) (3) ()))
    (check-equal? (tensor-dtype (caddr (buffers bn))) 'int64)
    (check-equal? (tensor->list (car (buffers bn))) '(0.0 0.0 0.0))
    (check-equal? (tensor->list (cadr (buffers bn))) '(1.0 1.0 1.0))
    (manual-seed! 0)
    (define x (add (mul (randn 4 3 5 5) 3.0) 2.0))
    (define y (bn x))
    (check-equal? (tensor-shape y) '(4 3 5 5))
    (define per-channel (reshape (transpose y 0 1) 3 100))
    (for ([c (in-range 3)])
      (check-= (item (mean (select per-channel 0 c))) 0.0 1e-5))
    (check-= (item (caddr (buffers bn))) 1 0)
    ;; momentum 0.1 of a batch mean near 2 and a batch variance near 9
    (for ([m (in-list (tensor->list (car (buffers bn))))])
      (check-true (< 0.1 m 0.3)))
    (for ([v (in-list (tensor->list (cadr (buffers bn))))])
      (check-true (< 1.5 v 2.1)))
    (define z (in-eval-mode bn (bn x)))
    (check-false (equal? (tensor->list z) (tensor->list y))
                 "eval normalizes with the running statistics")
    (check-= (item (caddr (buffers bn))) 1 0)
    (check-exn #rx"image-batch" (lambda () (bn (randn 4 3))))
    (check-equal? (object-name bn) 'BatchNorm2d))

  (test-case "BatchNorm1d layer: [N C] and [N C L] inputs"
    (define bn (BatchNorm1d 4 #:momentum 0.5 #:eps 1e-3))
    (check-true (batch-norm1d? bn))
    (check-equal? (tensor-shape (bn (randn 8 4))) '(8 4))
    (check-equal? (tensor-shape (bn (randn 8 4 6))) '(8 4 6))
    (check-= (item (caddr (buffers bn))) 2 0)
    (check-exn #rx"feature-batch" (lambda () (bn (randn 2 4 3 3))))
    (check-equal? (object-name bn) 'BatchNorm1d))

  (test-case "BatchNorm2d: gradients reach the affine parameters"
    (define bn (BatchNorm2d 2))
    (define x (randn 3 2 4 4))
    (backward! (mean (mul (bn x) (bn x))))
    (for ([p (in-list (parameters bn))])
      (check-true (has-grad? p))))

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

  (test-case "binary-cross-entropy-with-logits, huber-loss, l1-loss: known values"
    (define zero (tensor '(0.0 0.0)))
    (define labels (tensor '(1.0 0.0)))
    (define log2 0.6931472)
    (check-= (item (binary-cross-entropy-with-logits zero labels)) log2 1e-6)
    (check-= (item (binary-cross-entropy-with-logits
                    zero labels #:weight (tensor '(1.0 3.0))))
             (* 2 log2) 1e-6)
    (check-= (item (binary-cross-entropy-with-logits
                    zero labels #:pos-weight (tensor '(3.0 3.0))))
             (* 2 log2) 1e-6)
    (define x (tensor '(0.0 0.0 0.0)))
    (define target (tensor '(0.5 2.0 -3.0)))
    (check-= (item (huber-loss x target)) 1.375 1e-6)
    (check-= (item (huber-loss x target #:delta 2)) 2.0416667 1e-6)
    (check-= (item (l1-loss x target)) 1.8333333 1e-6)
    (check-exn exn:fail:contract? (lambda () (huber-loss x target #:delta 0))))

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

  (test-case "GroupNorm on mps: same values, gradient back on the device"
    (when (mps-available?)
      (manual-seed! 0)
      (define xs (tensor->list (randn 2 4 3 3)))
      (define (out+grad dev)
        (define x (to-device (reshape (tensor xs) 2 4 3 3) dev))
        (requires-grad! x)
        (define y ((to (GroupNorm 2 4) dev) x))
        (backward! (mean (mul y y)))
        (values y (grad x)))
      (define-values (cpu-y cpu-g) (out+grad 'cpu))
      (define-values (mps-y mps-g) (out+grad 'mps))
      (check-equal? (tensor-device mps-y) (mps-device))
      (check-equal? (tensor-device mps-g) (mps-device))
      (for ([a (in-list (tensor->list cpu-y))]
            [b (in-list (tensor->list (to-device mps-y 'cpu)))])
        (check-= a b 1e-5))
      (for ([a (in-list (tensor->list cpu-g))]
            [b (in-list (tensor->list (to-device mps-g 'cpu)))])
        (check-= a b 1e-5))))

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

  (test-case "ema: construction and the first update copy, later updates decay"
    (manual-seed! 0)
    (define net (Linear 2 2))
    (define avg (ema net (Linear 2 2) #:decay 0.5))
    (check-true (ema? avg))
    (check-equal? (ema-decay avg) 0.5)
    (define (weights layer) (tensor->list (car (parameters layer))))
    (check-equal? (weights (ema-average avg)) (weights net))
    (with-no-grad
      (for ([p (in-list (parameters net))])
        (mul! p 3.0)))
    (ema-update! avg)
    (check-equal? (weights (ema-average avg)) (weights net))
    (with-no-grad
      (for ([p (in-list (parameters net))])
        (mul! p 3.0)))
    (ema-update! avg)
    (for ([q (in-list (weights (ema-average avg)))]
          [p (in-list (weights net))])
      (check-= q (* p 2/3) 1e-6))
    (check-exn #rx"shape for shape" (lambda () (ema net (Linear 3 3))))
    (check-exn #rx"separate layer" (lambda () (ema net net)))
    (check-exn #rx"device and dtype" (lambda () (ema net (to (Linear 2 2) 'float64))))
    (define shared (Parameter (ones 3)))
    (define-layer holder (w) #:init (w0) (set! w w0) #:forward (x) x)
    (define aliased (ema (holder shared) (holder (Parameter shared))))
    (ema-update! aliased)
    (ema-update! aliased)
    (check-equal? (tensor->list shared) '(1.0 1.0 1.0)
                  "copying and averaging through shared storage leave the values intact")
    (to net 'float64)
    (to (ema-average avg) 'float64)
    (ema-update! avg)
    (check-equal? (tensor-dtype (car (parameters (ema-average avg)))) 'float64
                  "the cached weight follows the average across a move")
    (check-exn #rx"^ema: contract violation"
               (lambda () (ema net (Linear 2 2) #:decay 2))))

  (test-case "dropout: train drops/scales, eval is identity, mode recurses"
    (manual-seed! 0)
    (define d (Dropout #:p 0.5))
    (check-equal? (object-name d) 'Dropout)
    (check-equal? (tensor->list ((Dropout #:p 0) (ones 3))) '(1.0 1.0 1.0)
                  "an exact probability reaches the op as a flonum")
    (define x (ones 100))
    (define tr (tensor->list (d x)))
    (check-true (andmap (lambda (v) (or (= v 0.0) (= v 2.0))) tr))
    (check-true (> (length (filter zero? tr)) 0) "nothing was dropped")
    (eval! d)
    (check-equal? (tensor->list (d x)) (tensor->list x))
    (train! d)
    (check-true (andmap (lambda (v) (or (= v 0.0) (= v 2.0)))
                        (tensor->list (d x))))
    (check-equal? (state-dict d) '() "mode is not a state-dict entry")
    (define leaf (requires-grad! (ones 100)))
    (backward! (sum (d leaf)))
    (check-true (has-grad? leaf))
    (check-true (andmap (lambda (v) (or (= v 0.0) (= v 2.0)))
                        (tensor->list (grad leaf)))
                "the dropout mask flows back to the input"))

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
    (in-eval-mode lin (check-false (layer-training? lin)))
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
    (delete-file path))

  (test-case "BatchNorm2d: the running statistics and the counter round-trip"
    (define bn (BatchNorm2d 2))
    (bn (add (randn 3 2 4 4) 5.0))
    (define path (make-temporary-file "rkt-bn-~a.safetensors"))
    (save-state! bn path)
    (define bn2 (BatchNorm2d 2))
    (load-state! bn2 path)
    (check-equal? (map car (state-dict bn2))
                  '("weight" "bias" "running-mean" "running-var"
                    "num-batches-tracked"))
    (for ([a (in-list (state-dict bn))] [b (in-list (state-dict bn2))])
      (check-equal? (tensor->list (cdr a)) (tensor->list (cdr b))))
    (check-equal? (tensor-dtype (caddr (buffers bn2))) 'int64)
    (check-= (item (caddr (buffers bn2))) 1 0)
    (delete-file path))

  (define-layer Typed (w f16 bf16 u8 f64 mask)
    #:init ()
    (set! w (Parameter (tensor '(0.5 -1.5))))
    (set! f16 (Buffer (tensor '(1.0 0.1 -2.0) #:dtype 'float16)))
    (set! bf16 (Buffer (to-dtype (tensor '((1.0 0.1) (3.0 4.0))) 'bfloat16)))
    (set! u8 (Buffer (tensor (bytes 0 9 255))))
    (set! f64 (Buffer (to (tensor '(1.0 2.0)) 'float64)))
    (set! mask (Buffer (gt (tensor '(1.0 -1.0 1.0)) 0)))
    #:forward (x) x)

  (define-layer Wide (w f16 bf16 u8 f64 mask)
    #:init ()
    (set! w (Parameter (zeros 2)))
    (set! f16 (Buffer (zeros 3)))
    (set! bf16 (Buffer (zeros 2 2)))
    (set! u8 (Buffer (zeros 3)))
    (set! f64 (Buffer (zeros 2)))
    (set! mask (Buffer (zeros 3)))
    #:forward (x) x)

  (test-case "safetensors carries every dtype: the half pair, uint8, float64"
    (define a (Typed))
    (define path (make-temporary-file "rkt-typed-~a.safetensors"))
    (save-state! a path)
    (define header-len (integer-bytes->integer (file->bytes path) #f #f 0 8))
    (define header
      (bytes->string/utf-8 (subbytes (file->bytes path) 8 (+ 8 header-len))))
    (for ([tag (in-list '("F32" "F16" "BF16" "U8" "F64" "BOOL"))])
      (check-true (regexp-match? (regexp-quote tag) header) tag))
    (define b (Typed))
    (with-no-grad
      (for ([t (in-list (append (parameters b) (buffers b)))])
        (copy! t (zeros-like t))))
    (load-state! b path)
    (for ([x (in-list (state-dict a))] [y (in-list (state-dict b))])
      (check-equal? (tensor-dtype (cdr x)) (tensor-dtype (cdr y)) (car x))
      (check-equal? (tensor->list (cdr x)) (tensor->list (cdr y)) (car x)))
    ;; a file in one dtype loads into a model in another: copy! converts
    (define c (Wide))
    (load-state! c path)
    (check-equal? (map (lambda (e) (tensor-dtype (cdr e))) (state-dict c))
                  '(float32 float32 float32 float32 float32 float32))
    (check-equal? (tensor->list (cdr (assoc "u8" (state-dict c))))
                  '(0.0 9.0 255.0))
    (delete-file path)))
