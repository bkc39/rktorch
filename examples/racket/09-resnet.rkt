#lang scribble/lp2

@(require (for-label (except-in racket/base abs cos exp log sin sort sqrt max min length + - * /)
                     torch torch/nn torch/data/loader torch/vision/cifar10
                     torch/vision/resnet torch/vision/transforms))

@title[#:tag "ex-resnet"]{Training a ResNet classifier on CIFAR-10}

The classic supervised-vision recipe: a ResNet-18 shaped for 32x32 images,
random crops and flips on the device, SGD with momentum and weight decay
under a one-cycle learning-rate schedule, and the forward pass in
@racket[bfloat16] under @racket[with-autocast] on the GPU. The network lives
in @racketmodname[torch/vision/resnet]; the augmentation in
@racketmodname[torch/vision/transforms]; this program is the loop around
them, the same shape as the MNIST convnet's with three more ingredients.

@chunk[<r09-require>
(require torch torch/nn torch/data/loader
         (only-in torch/vision/cifar10
                  cifar10-dataset load-cifar10 load-cifar10-fixture)
         torch/vision/resnet
         torch/vision/transforms)]

@chunk[<r09-provide>
(provide pick-device accuracy train-step run-example train-cifar10)]

@bold{Accuracy.} The test set in slices of 500 under @racket[in-eval-mode],
so the batch-norm layers normalise with their running statistics, and
@racket[with-no-grad], so no graph is built.

@chunk[<r09-accuracy>
(define (pick-device)
  (accelerator-if-available))

(define (accuracy net xs ys)
  (in-eval-mode net
    (with-no-grad
      (define n (car (tensor-shape xs)))
      (define correct
        (for/sum ([start (in-range 0 n 500)])
          (define len (min 500 (- n start)))
          (define preds (argmax (net (narrow xs 0 start len)) 1))
          (item (sum (eq preds (narrow ys 0 start len))))))
      (exact->inexact (/ correct n)))))]

@bold{One step.} Cross-entropy on the logits, the backward pass outside the
autocast extent as PyTorch recommends, then the optimizer and, when there
is one, the schedule. Autocast is asked for only on CUDA, judged from the
batch's own device: the CPU arm of the tests and the parity twin train in
float32.

@chunk[<r09-step>
(define (train-step net opt xs ys #:schedule [schedule #f])
  (zero-grads! opt)
  (define device (tensor-device xs))
  (define loss
    (if (eq? (device-type device) 'cuda)
        (with-autocast #:device device (cross-entropy (net xs) ys))
        (cross-entropy (net xs) ys)))
  (backward! loss)
  (step! opt)
  (when schedule (step! schedule))
  (item loss))]

@bold{The deterministic core.} @racket[run-example] is the seeded, offline
entry the test harness and the parity twin drive: a narrow ResNet, base
width 16, three full-batch SGD steps with momentum and weight decay on the
committed 256-image fixture, no augmentation and no autocast, and the
per-step losses back. The rate is a tenth of the headline run's: batch
norm divides by the batch's own statistics, which amplifies the last-bit
differences between two torch builds' kernels, and at the full rate the
two trajectories part after two steps. @racket[#:batch] trains on a prefix
of the fixture, which the tests use to drive every kernel without the full
batch's activations.

@chunk[<r09-run>
(define (run-example #:steps [steps 3] #:batch [batch #f]
                     #:device [device (pick-device)])
  (with-default-device device
    (manual-seed! 0)
    (define-values (all-xs all-ys) (load-cifar10-fixture))
    (define xs (if batch (narrow all-xs 0 0 batch) all-xs))
    (define ys (if batch (narrow all-ys 0 0 batch) all-ys))
    (define net (ResNet #:base 16))
    (define opt (sgd (parameters net) #:lr 0.005 #:momentum 0.9
                     #:weight-decay 5e-4))
    (define losses
      (for/list ([_ (in-range steps)])
        (zero-grads! opt)
        (define loss (cross-entropy (net xs) ys))
        (backward! loss)
        (step! opt)
        (item loss)))
    (values losses net device)))]

@bold{The headline run.} The full training set through the shuffled loader,
device resident; every batch cropped with four pixels of padding and
flipped at random on the device, the draws seeded through the loader's
generator; the ResNet-18 at its full width; SGD with Nesterov momentum
0.9 and weight decay 5e-4 under a one-cycle schedule peaking at the given
rate, stepped once per batch; and the test accuracy after every epoch.
On an RTX 3090 Ti an epoch takes about ten seconds in bfloat16, and thirty
epochs reach 93 to 94 percent.

@chunk[<r09-train>
(define (train-cifar10 #:epochs [epochs 30] #:batch [batch 128]
                       #:max-lr [max-lr 0.1]
                       #:device [device (pick-device)])
  (with-default-device device
    (manual-seed! 0)
    (define g (make-generator 0))
    (define loader
      (dataloader (cifar10-dataset 'train #:device device)
                  #:batch-size batch #:shuffle? #t #:generator g))
    (define-values (test-x test-y) (load-cifar10 'test #:device device))
    (define net (ResNet))
    (define opt (sgd (parameters net) #:lr max-lr #:momentum 0.9
                     #:nesterov? #t #:weight-decay 5e-4))
    (define schedule
      (one-cycle-lr opt #:max-lr max-lr
                    #:total-steps (* epochs (dataloader-length loader))))
    (for/list ([epoch (in-range epochs)])
      (for ([(xb yb) (in-dataloader loader)])
        (define augmented
          (random-horizontal-flip (random-crop xb #:padding 4 #:generator g)
                                  #:generator g))
        (train-step net opt augmented yb #:schedule schedule))
      (accuracy net test-x test-y))))]

@chunk[<*>
  <r09-require>
  <r09-provide>
  <r09-accuracy>
  <r09-step>
  <r09-run>
  <r09-train>]
