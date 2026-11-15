#lang scribble/lp2

@(require (only-in racket/format ~a ~r)
          (only-in racket/match match-define)
          torch/scribblings/labels
          (for-label torch/data/loader torch/vision/fashion-mnist))

@(define results
   (call-with-input-file
     (collection-file-path "fashion-mnist.rktd" "torch" "scribblings" "results")
     read))

@(define (percent x) (~r (* 100 x) #:precision '(= 1)))

@title[#:tag "ex-fashion-mnist"]{Fashion-MNIST, step for step with Lightning}

Zalando's Fashion-MNIST is MNIST's shape with harder pictures: shirts, sandals and
bags in place of digits. This chapter trains a small convnet on it with
@racket[fit] and every option a real recipe sets at once: AdamW with the norms and
biases left undecayed, a warmup into a cosine schedule stepped per optimizer step,
gradients accumulated over two batches and clipped, and mixed precision. Its twin,
@filepath{examples/python/20_fashion_mnist.py}, is the same program in Lightning
2.6.5, and the tests hold the two to each other step for step.

@chunk[<r20-require>
(require (only-in racket/match match)
         (only-in racket/math exact-ceiling)
         torch torch/nn torch/data/loader torch/train
         (only-in torch/vision/fashion-mnist
                  fashion-mnist-dataset load-fashion-mnist-fixture))]

@chunk[<r20-provide>
(provide FashionClassifier fashion-net pick-device precision-for seed batch-size accumulate
         clip-val lr weight-decay run-example train-fashion-mnist)]

@bold{The model.} Two convolution blocks, each a 3 by 3 convolution, a batch norm,
a ReLU and a 2 by 2 max pool, take the 28 by 28 picture to 32 channels of 7 by 7,
and two linear layers classify it. There is no dropout, whose draws the twin could
not repeat.

@chunk[<r20-model>
(define-layer conv-block (conv bn pool)
  #:init (in out)
  (set! conv (Conv2d in out 3 #:padding 1))
  (set! bn (BatchNorm2d out))
  (set! pool (MaxPool2d 2))
  #:forward (x)
  (~> x conv bn relu pool))

(define (fashion-net)
  (Sequential (conv-block 1 16) (conv-block 16 32) (Flatten) (Linear 1568 128) relu
              (Linear 128 10)))]

@bold{The trainer.} @racket[weight-decay-param-groups] splits the parameters in
two, the weights that decay and the biases and norms that do not, and
@racket[adamw] takes the groups. The schedule rises linearly from a tenth of the
rate over the first quarter of the optimizer steps and then follows a cosine down;
@racket[run-estimated-stepping-batches] is how many steps the fit will take, as
Lightning's @tt{estimated_stepping_batches}.

@chunk[<r20-trainer>
(define seed 0)
(define batch-size (hasheq 'fixture 32 'full 128))
(define accumulate 2)
(define clip-val 0.5)
(define lr 2e-3)
(define weight-decay 0.05)

(define-trainer FashionClassifier (net)
  (define-init () (set! net (fashion-net)))
  (define-forward (x) (net x))
  (define-optimizers
    (define opt (adamw (weight-decay-param-groups net #:weight-decay weight-decay) #:lr lr))
    (define total (run-estimated-stepping-batches run))
    (define warmup (exact-ceiling (/ total 4)))
    (values opt
            (scheduler-config
             (sequential-lr opt
                            (list (linear-lr opt #:start-factor 0.1 #:total-iters warmup)
                                  (cosine-annealing-lr opt #:t-max (- total warmup)))
                            #:milestones (list warmup))
             #:interval 'step)))
  (define-training-step (xb yb)
    (define loss (cross-entropy (net xb) yb))
    (log! 'train/loss loss #:on-epoch? #t)
    loss)
  (define-validation-step (xb yb)
    (define logits (net xb))
    (log! 'val/loss (cross-entropy logits yb))
    (log! 'val/acc (~> logits (argmax 1) (eq yb) (to-dtype 'float32) mean))))]

@bold{Device and precision.} On a GPU the run is @racket['16-mixed], half
precision with a @racket[grad-scaler]; elsewhere it is full precision.

@chunk[<r20-device>
(define (pick-device)
  (accelerator-if-available))

(define (precision-for device)
  (match (device-type device)
    ['cuda '16-mixed]
    [_ '32-true]))

(define (device-of d)
  (match d
    ['cpu (cpu-device)]
    ['cuda (cuda-device)]
    [_ d]))]

@bold{The deterministic core.} @racket[run-example] is the seeded, offline run the
tests and the twin drive: two epochs on the committed fixture, 512 training and
128 test pictures, in shuffled batches of 32, so 16 optimizer steps of which 4 are
warmup. It answers the run and the trained classifier.

@chunk[<r20-run>
(define (run-example #:device [device 'cpu] #:precision [precision '32-true]
                     #:epochs [epochs 2] #:callbacks [callbacks '()] #:logger [logger #f])
  (error 'run-example "unimplemented"))]

@bold{The headline run.} The full sets, resident on the device, in batches of 128,
and the test set after every epoch. @racket[train-fashion-mnist] answers one row
per epoch, the epoch, the mean training loss, the test loss and accuracy and the
seconds it took, and the run.

@chunk[<r20-train>
(define (train-fashion-mnist #:epochs [epochs 5] #:device [device (pick-device)]
                             #:precision [precision (precision-for device)]
                             #:callbacks [callbacks '()])
  (error 'train-fashion-mnist "unimplemented"))]

@bold{The result.} @filepath{scripts/fashion-mnist-results.rkt} ran this program and
its twin at commit @tt{@(hash-ref results 'commit)} on @(hash-ref results 'hardware)
on @(hash-ref results 'date), @(~a (length (hash-ref (hash-ref results 'racket) 'epochs)))
epochs at precision @tt{@(hash-ref results 'precision)}, seed
@(~a (hash-ref results 'seed)). Test accuracy after the last epoch is
@(percent (hash-ref (hash-ref results 'racket) 'accuracy)) percent here and
@(percent (hash-ref (hash-ref results 'lightning) 'accuracy)) percent in Lightning
@(hash-ref results 'lightning-version).

@(tabular
  #:style 'boxed
  #:sep (hspace 2)
  (cons (list (bold "epoch") (bold "side") (bold "train loss") (bold "test loss")
              (bold "test accuracy") (bold "seconds"))
        (for*/list ([i (in-range (length (hash-ref (hash-ref results 'racket) 'epochs)))]
                    [side (in-list '(racket lightning))])
          (match-define (list epoch train-loss val-loss acc seconds)
            (list-ref (hash-ref (hash-ref results side) 'epochs) i))
          (list (~a epoch)
                (if (eq? side 'racket) "Racket" "Lightning")
                (~r train-loss #:precision '(= 4))
                (~r val-loss #:precision '(= 4))
                (string-append (percent acc) "%")
                (~r seconds #:precision '(= 1))))))

@chunk[<*>
  <r20-require>
  <r20-provide>
  <r20-model>
  <r20-trainer>
  <r20-device>
  <r20-run>
  <r20-train>]
