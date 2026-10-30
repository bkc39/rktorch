#lang scribble/lp2

@(require torch/scribblings/labels
          (for-label torch/data/loader torch/vision/augment
                     torch/vision/cifar10 torch/vision/vit))

@title[#:tag "ex-vit"]{Training a Vision Transformer on CIFAR-10}

The Vision Transformer of Dosovitskiy et al. (@italic{An Image is Worth
16x16 Words}, ICLR 2021) reads an image as a sequence: it cuts the picture
into square patches and runs them through the encoder of a language model.
This chapter trains a small one from scratch with @racket[fit]. The network
is @racket[VisionTransformer] from @racketmodname[torch/vision/vit], the
augmentation is a @racket[crop-view] from
@racketmodname[torch/vision/augment], and this program is the trainer around
them, the shape of @secref["ex-mnist"]'s with an augmentation hook and a
schedule.

@chunk[<r18-require>
(require racket/match
         torch torch/nn torch/data/loader torch/train
         (only-in torch/vision/augment crop-view)
         (only-in torch/vision/cifar10 cifar10-dataset load-cifar10-fixture)
         (only-in torch/vision/vit VisionTransformer))]

@chunk[<r18-provide>
(provide CifarViT augment-batch cifar-vit pick-device run-example small-vit
         train-cifar10)]

@bold{The model.} A @racket[Conv2d] with a patch-wide kernel and stride cuts
the image and projects each patch to a token in one call. The paper's 16
pixels would leave four tokens of a 32x32 image; patches of 4 leave 64.
Attention does not see order, so a learned position is added to each token.
A learned class token goes in front, attends to the patches through six
blocks, and is all the head reads. @racket[small-vit] is the network the
tests and the parity twin train: narrower, shallower, and without dropout,
whose draws the twin could not repeat.

@chunk[<r18-model>
(define (cifar-vit #:width [width 256] #:depth [depth 6] #:heads [heads 8]
                   #:dropout [dropout 0.1])
  (VisionTransformer #:image-size 32 #:patch 4 #:width width #:depth depth
                     #:heads heads #:dropout dropout #:classes 10))

(define (small-vit)
  (cifar-vit #:width 64 #:depth 2 #:heads 4 #:dropout 0.0))]

@bold{Augmentation.} A convolution assumes neighbouring pixels matter and
that a shape means the same thing anywhere in the image. A ViT assumes
neither and has to learn both, which on 50,000 images it can only do from
more varied images. A @racket[crop-view] makes them on the device, a whole
batch at a time: a crop of at least half the area at a random aspect ratio
resized back to 32, a coin-flip mirror, colour jitter on half the images and
grayscale on a tenth. The loader's images lie in -1 to 1 and a view takes 0
to 1, so @racket[augment-batch] shifts them in and the view's
@racket[#:mean] and @racket[#:std] shift them back. The draws come from the
device's default stream, which @racket[manual-seed!] seeds.

@chunk[<r18-augment>
(define view
  (crop-view #:size 32 #:scale '(0.5 1.0)
             #:jitter '(0.4 0.4 0.4 0.1) #:jitter-p 0.5 #:gray-p 0.1
             #:mean '(0.5 0.5 0.5) #:std '(0.5 0.5 0.5)))

(define (augment-batch xb)
  (~> xb (+ 1) (/ 2) view))]

@bold{The trainer.} @racket[adam] with a small weight decay, the L2 term in
the gradient rather than AdamW's decoupled one. @racket[one-cycle-lr] is
warmup and cosine in one schedule: a tenth of the steps rising to the peak
rate, the rest falling to almost nothing. @racket[scheduler-config] steps it
per batch, so the trainer has to be told how many steps there are.
@racket[on-after-batch-transfer] runs when a batch reaches the device; the
stage keeps the test images as they are.

@chunk[<r18-trainer>
(define-trainer CifarViT (net augment max-lr total-steps)
  (define-init (vit #:total-steps steps #:max-lr [lr 1e-3]
                    #:augment [aug augment-batch])
    (set! net vit)
    (set! augment aug)
    (set! max-lr lr)
    (set! total-steps steps))
  (define-forward (x) (net x))
  (define-optimizers
    (define opt (adam (parameters net) #:lr max-lr #:weight-decay 1e-4))
    (values opt
            (scheduler-config
             (one-cycle-lr opt #:max-lr max-lr #:total-steps total-steps
                           #:pct-start 0.1)
             #:interval 'step)))
  (define-hook (on-after-batch-transfer xb yb)
    (match (run-stage run)
      ['train (values (augment xb) yb)]
      [_ (values xb yb)]))
  (define-training-step (xb yb)
    (define loss (cross-entropy (net xb) yb))
    (log! 'train/loss loss #:on-epoch? #t)
    loss)
  (define-validation-step (xb yb)
    (log! 'val/acc (~> (net xb) (argmax 1) (eq yb) (to-dtype 'float32) mean))))]

@chunk[<r18-device>
(define (pick-device)
  (accelerator-if-available))]

@bold{The deterministic core.} @racket[run-example] is the seeded, offline
entry the tests and the parity twin drive: @racket[small-vit], no
augmentation, and twenty full-batch steps on the committed 256-image
fixture, enough for the schedule's one warmup step and its decay. It answers
the per-step losses, the trained classifier and the device. The twin
@filepath{examples/python/18_vit.py} starts from the weights this network
saved; its losses and final weights agree within @tt{1e-4}.

@chunk[<r18-run>
(define (run-example #:steps [steps 20] #:device [device (pick-device)])
  (with-default-device device
    (manual-seed! 0)
    (define-values (xs ys) (load-cifar10-fixture))
    (define classifier
      (CifarViT (small-vit) #:total-steps steps #:augment values))
    (define losses '())
    (define (record! _r loss _extra _xb _yb)
      (set! losses (cons (item loss) losses)))
    (fit classifier
         (dataloader (tensor-dataset xs ys) #:batch-size (length xs))
         #:max-steps steps #:logger #f
         #:callbacks (list (lambda-callback #:on-train-batch-end record!)))
    (values (reverse losses) classifier device)))]

@bold{The headline run.} The full training set, device resident, in seeded
shuffled batches of 256, and the test set after every epoch. A
@racket[lambda-callback] keeps one row per epoch: the mean training loss,
the test accuracy and the seconds it took.

@chunk[<r18-train>
(define (train-cifar10 #:epochs [epochs 300] #:batch [batch 256]
                       #:device [device (pick-device)]
                       #:train [train-set (cifar10-dataset 'train #:device device)]
                       #:test [test-set (cifar10-dataset 'test #:device device)])
  (with-default-device device
    (manual-seed! 0)
    (define loader
      (dataloader train-set #:batch-size batch #:shuffle? #t
                  #:generator (make-generator 0)))
    (define rows '())
    (define clock (current-inexact-milliseconds))
    (define (record! r)
      (define now (current-inexact-milliseconds))
      (set! rows (cons (list (add1 (run-epoch r))
                             (run-metric r 'train/loss)
                             (run-metric r 'val/acc)
                             (/ (- now clock) 1000.0))
                       rows))
      (set! clock now))
    (fit (CifarViT (cifar-vit)
                   #:total-steps (* epochs (dataloader-length loader)))
         loader
         #:val (dataloader test-set #:batch-size 500)
         #:max-epochs epochs #:num-sanity-val-steps 0
         #:callbacks (list (lambda-callback #:on-train-epoch-end record!)))
    (reverse rows)))]

@chunk[<*>
  <r18-require>
  <r18-provide>
  <r18-model>
  <r18-augment>
  <r18-trainer>
  <r18-device>
  <r18-run>
  <r18-train>]
