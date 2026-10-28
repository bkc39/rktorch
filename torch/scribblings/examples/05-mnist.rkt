#lang scribble/lp2

@(require torch/scribblings/labels
          (for-label torch/data/loader))

@title[#:tag "ex-mnist"]{Training a convnet on MNIST}

The v2 capstone: a LeNet-ish convolutional network --- two
@racket[Conv2d]/@racket[max-pool2d] stages into two @racket[Linear] layers ---
trained on the handwritten-digit dataset with the @racket[adam] optimizer by
@racket[fit], the trainer of @secref["train"]. The same code path runs on the
GPU when one is present and on the CPU otherwise; on an RTX 3090 Ti it reaches
~98% test accuracy in three epochs, and ~97% in a single CPU epoch.

The @emph{layer} constructors are PascalCase (@racket[Conv2d], @racket[Linear],
@racket[MaxPool2d], @racket[Flatten]), mirroring PyTorch's @tt{torch.nn}
classes; the @emph{functional} ops stay lowercase on @racketmodname[torch]
(@racket[relu], @racket[max-pool2d], like @tt{torch.relu}). Because the two casings differ,
@racket[(require torch torch/nn)] never collides (#11) --- no prefix or
@racket[except-in] needed.

@chunk[<r05-require>
(require torch torch/nn torch/data/loader torch/train
         (only-in torch/vision/mnist load-mnist load-mnist-fixture))]

@chunk[<r05-provide>
(provide MnistClassifier accuracy convnet pick-device run-example train-mnist)]

@bold{The model.} A @racket[conv-block] bundles a convolution with the
activation and pooling that follow it, so the network is a
@racket[Sequential] of two blocks, a @racket[Flatten] and the classifier,
with the plain @racket[relu] between the dense layers a step like any other.
Nothing wraps the @racket[Sequential]: its forward is the whole forward, so
@racket[convnet] is a function that builds a fresh one, and the parameters
are named by step, as @racket["0.conv.weight"]. The spatial arithmetic is the usual @tt{valid}-convolution bookkeeping:
@tt{28 -c3-> 26 -pool-> 13 -c3-> 11 -pool-> 5}, so the flattened feature map is
@tt{32*5*5 = 800} wide going into the first dense layer.

@chunk[<r05-model>
(define-layer conv-block (conv pool)
  #:init (in-channels out-channels)
  (set! conv (Conv2d in-channels out-channels 3))
  (set! pool (MaxPool2d 2))
  #:forward (x)
  (~> x conv relu pool))

(define (convnet)
  (Sequential (conv-block 1 16) (conv-block 16 32) (Flatten)
              (Linear 800 128) relu (Linear 128 10)))]

@bold{The trainer.} @racket[define-trainer] holds the method: the network, its
optimizer, and what one training and one validation batch do. The training
step answers the loss, and @racket[fit] runs the backward pass and the
@racket[adam] step around it. The validation step logs the batch's accuracy;
@racket[fit] runs it in @racket[eval!] mode under @racket[with-no-grad] and
averages it over the epoch. The forward makes the trainer a classifier in its
own right.

@chunk[<r05-trainer>
(define-trainer MnistClassifier (net)
  (define-init () (set! net (convnet)))
  (define-forward (x) (net x))
  (define-optimizers (adam (parameters net) #:lr 0.001))
  (define-training-step (xb yb)
    (define loss (cross-entropy (net xb) yb))
    (log! 'train/loss loss)
    loss)
  (define-validation-step (xb yb)
    (log! 'val/acc (~> (net xb) (argmax 1) (eq yb) (to-dtype 'float32) mean))))]

@bold{The device.} Pick the accelerator as the MLP example does; setting it as
the process default means every tensor built afterwards --- the model's
parameters and each batch alike --- lands there, so the whole loop follows.

@chunk[<r05-device>
(define (pick-device)
  (accelerator-if-available))]

@bold{Accuracy outside the trainer.} For a net trained some other way, as in
@secref["ex-workers"]: @racket[in-eval-mode] and @racket[with-no-grad] around
batches of 1000, so the test set never has to live on the device all at once.
@racket[in-eval-mode] restores the prior mode even if a batch raises.

@chunk[<r05-accuracy>
(define (accuracy net xs ys)
  (in-eval-mode net
    (with-no-grad
      (define n (car (tensor-shape xs)))
      (define correct
        (for/sum ([start (in-range 0 n 1000)])
          (define len (min 1000 (- n start)))
          (define preds (argmax (net (narrow xs 0 start len)) 1))
          (item (sum (eq preds (narrow ys 0 start len))))))
      (exact->inexact (/ correct n)))))]

@bold{The deterministic core.} @racket[run-example] is the seeded, offline entry
the test harness drives: it fits a fresh @racket[MnistClassifier] for
@racket[steps] full-batch steps on the committed 256-image fixture and returns
the per-step losses, read by a @racket[lambda-callback], the trained classifier,
and the device. Full-batch keeps it reproducible across languages: with the
same seed the @racket[Conv2d] and @racket[Linear] inits draw value-for-value
like @tt{nn.Conv2d}/@tt{nn.Linear}, and the identical updates track
@tt{torch.optim.Adam}. @racket[with-default-device] sets the process default
device for the dynamic extent of the run and restores the prior one, even if a
step raises.

@chunk[<r05-run>
(define (run-example #:steps [steps 5] #:device [device (pick-device)])
  (with-default-device device
    (manual-seed! 0)
    (define-values (xs ys) (load-mnist-fixture))
    (define classifier (MnistClassifier))
    (define losses '())
    (define (record! r loss extra . batch)
      (set! losses (cons (item loss) losses)))
    (fit classifier
         (dataloader (tensor-dataset xs ys) #:batch-size (car (shape xs)))
         #:max-steps steps #:logger #f
         #:callbacks (list (lambda-callback #:on-train-batch-end record!)))
    (values (reverse losses) classifier device)))]

@bold{The real thing.} @racket[train-mnist] is the headline run: it downloads the
full dataset (cached under @envvar{RKTORCH_MNIST_DIR} or the system cache dir),
fits for @racket[epochs] epochs of shuffled minibatches drawn by a
@racket[dataloader] over the device-resident training set (a seeded
@racket[make-generator], so the batch order replays), validates on the test
set after each, and answers the last epoch's test accuracy. The default
@racket[stdout-logger] prints the training loss every 50 steps and the
accuracy at each epoch's end.

@chunk[<r05-train-mnist>
(define (train-mnist #:epochs [epochs 3] #:batch [batch 128]
                     #:device [device (pick-device)])
  (with-default-device device
    (manual-seed! 0)
    (define-values (train-x train-y) (load-mnist 'train))
    (define-values (test-x test-y) (load-mnist 'test))
    (define r
      (fit (MnistClassifier)
           (dataloader (tensor-dataset train-x train-y)
                       #:batch-size batch #:shuffle? #t
                       #:generator (make-generator 0))
           #:val (dataloader (tensor-dataset test-x test-y) #:batch-size 1000)
           #:max-epochs epochs))
    (run-metric r 'val/acc)))]

@chunk[<*>
  <r05-require>
  <r05-provide>
  <r05-model>
  <r05-trainer>
  <r05-device>
  <r05-accuracy>
  <r05-run>
  <r05-train-mnist>]
