#lang scribble/lp2

@(require (only-in racket/format ~a ~r)
          (only-in racket/match match-define)
          torch/scribblings/labels
          (for-label torch/data/loader torch/vision/augment
                     torch/vision/cifar10 torch/vision/vit))

@(define results
   (call-with-input-file
     (collection-file-path "vit.rktd" "torch" "scribblings" "results")
     read))

@(define (percent x) (~r (* 100 x) #:precision '(= 1)))

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
(require (only-in racket/match match match-define)
         torch torch/nn torch/data/loader torch/train
         (only-in torch/vision/augment crop-view)
         (only-in torch/vision/cifar10 cifar10-dataset load-cifar10-fixture)
         (only-in torch/vision/vit VisionTransformer))]

@chunk[<r18-provide>
(provide CifarViT augment-batch cifar-vit pick-device precision run-example
         seed small-vit train-cifar10 vit-config)]

@bold{The model.} @racket[VisionTransformer] cuts the image into square
patches and projects each to a token, adds a learned position to each token,
since attention does not see order, and puts a learned class token in front,
the only token the head reads; its reference entry has the rest. Patches of
4 leave 64 tokens of a 32x32 image, where the paper's 16 would leave four.
@racket[vit-config] is the network this chapter trains. @racket[small-vit]
is the one the tests and the parity twin train: narrower, shallower, and
without dropout, whose draws the twin could not repeat.

@chunk[<r18-model>
(define vit-config
  (hasheq 'patch 4 'width 256 'depth 6 'heads 8 'dropout 0.1))

(define (cifar-vit [config vit-config])
  (match-define (hash-table ('patch patch) ('width width) ('depth depth)
                            ('heads heads) ('dropout dropout))
    config)
  (VisionTransformer #:image-size 32 #:patch patch #:width width #:depth depth
                     #:heads heads #:dropout dropout #:classes 10))

(define (small-vit)
  (cifar-vit (hash-set* vit-config 'width 64 'depth 2 'heads 4 'dropout 0.0)))]

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
(define cropped
  (crop-view #:size 32 #:scale '(0.5 1.0)
             #:jitter '(0.4 0.4 0.4 0.1) #:jitter-p 0.5 #:gray-p 0.1
             #:mean '(0.5 0.5 0.5) #:std '(0.5 0.5 0.5)))

(define (augment-batch xb)
  (~> xb (+ 1) (/ 2) cropped))]

@bold{The trainer.} @racket[adam] with a small weight decay, the L2 term in
the gradient rather than AdamW's decoupled one. @racket[one-cycle-lr] is
warmup and cosine in one schedule: a tenth of the steps rising to the peak
rate, the rest falling to almost nothing. @racket[scheduler-config] steps it
per batch, so the trainer has to be told how many steps there are.
@racket[on-after-batch-transfer] runs when a batch reaches the device; the
stage keeps the test images as they are.

@chunk[<r18-trainer>
(define-trainer CifarViT (net #:total-steps total-steps #:max-lr [max-lr 1e-3]
                          #:augment [augment augment-batch])
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
saved; the parity test holds its losses and final weights to this run's.

@chunk[<r18-run>
(define (run-example #:steps [steps 20] #:device [device (pick-device)])
  (with-default-device device
    (manual-seed! seed)
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
the test accuracy and the seconds it took. @racket[train-cifar10] answers
the rows and the trained classifier.

@chunk[<r18-train>
(define seed 0)
(define precision '32-true)

(define (train-cifar10 #:epochs [epochs 300] #:batch [batch 256]
                       #:device [device (pick-device)]
                       #:train [train-set
                                (cifar10-dataset 'train #:device device)]
                       #:test [test-set
                               (cifar10-dataset 'test #:device device)]
                       #:callbacks [callbacks '()])
  (with-default-device device
    (manual-seed! seed)
    (define loader
      (dataloader train-set #:batch-size batch #:shuffle? #t
                  #:generator (make-generator seed)))
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
    (define classifier
      (CifarViT (cifar-vit)
                #:total-steps (* epochs (dataloader-length loader))))
    (fit classifier
         loader
         #:val (dataloader test-set #:batch-size 500)
         #:max-epochs epochs #:num-sanity-val-steps 0 #:precision precision
         #:callbacks
         (cons (lambda-callback #:on-train-epoch-end record!) callbacks))
    (values (reverse rows) classifier)))]

@bold{The result.} @filepath{scripts/vit-results.rkt} ran this program at
commit @tt{@(hash-ref results 'commit)} on @(hash-ref results 'hardware)
on @(hash-ref results 'date): @(~a (length (hash-ref results 'epochs)))
epochs in @(~r (/ (hash-ref results 'seconds) 60) #:precision 0) minutes,
at precision @tt{@(hash-ref results 'precision)}, seed
@(~a (hash-ref results 'seed)). Test accuracy after the last epoch is
@(percent (hash-ref results 'accuracy)) percent; the best epoch reached
@(percent (hash-ref results 'best)), but the test set picks no epoch here.
The ResNet-18 of @secref["ex-resnet"] starts from the prior this network has
to learn: its convolutions assume locality and translation from the first
step.

@(tabular
  #:style 'boxed
  #:sep (hspace 2)
  (cons (list (bold "epoch") (bold "train loss") (bold "test accuracy")
              (bold "seconds"))
        (for/list ([row (in-list (hash-ref results 'epochs))]
                   #:do [(match-define (list epoch loss acc seconds) row)]
                   #:when (or (= 1 epoch) (zero? (modulo epoch 30))))
          (list (~a epoch)
                (~r loss #:precision '(= 4))
                (string-append (percent acc) "%")
                (~r seconds #:precision '(= 1))))))

@chunk[<*>
  <r18-require>
  <r18-provide>
  <r18-model>
  <r18-augment>
  <r18-trainer>
  <r18-device>
  <r18-run>
  <r18-train>]
