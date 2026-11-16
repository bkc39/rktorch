#lang scribble/lp2

@(require (only-in racket/file file->value)
          (only-in racket/format ~a)
          (only-in racket/match match-define)
          torch/scribblings/labels
          (for-label torch/data/loader torch/vision/fashion-mnist))

@(match-define (hash-table ('racket ours) ('lightning theirs) ('commit commit)
                           ('hardware hardware) ('date run-date) ('precision precision)
                           ('seed run-seed) ('lightning-version lightning-version)
                           ('python-torch python-torch))
   (file->value (collection-file-path "fashion-mnist.rktd" "torch" "scribblings" "results")))

@(define (percent x) (real->decimal-string (* 100 x) 1))

@title[#:tag "ex-fashion-mnist"]{Fashion-MNIST, step for step with Lightning}

Zalando's Fashion-MNIST is MNIST's shape with harder pictures: shirts, sandals and
bags in place of digits. This chapter trains a small convnet on it with
@racket[fit] and every option a real recipe sets at once: AdamW with the norms and
biases left undecayed, a warmup into a cosine schedule stepped per optimizer step,
gradients accumulated over two batches and clipped, and mixed precision. Its twin,
@filepath{examples/python/20_fashion_mnist.py}, is the same program in Lightning
2.6.5, and the tests hold the two to each other step for step.

@chunk[<r20-require>
(require (only-in racket/match match-define)
         (only-in racket/math exact-ceiling)
         torch torch/nn torch/data/loader torch/train)]

@chunk[<r20-provide>
(provide FashionClassifier fashion-net fashion-recipe train-fashion-mnist warmup-steps)]

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
(define fashion-recipe
  (hasheq 'seed 0 'lr 2e-3 'weight-decay 0.05 'accumulate 2 'clip 0.5))

(match-define (hash-table ('seed seed) ('lr lr) ('weight-decay weight-decay)
                          ('accumulate accumulate) ('clip clip-val))
  fashion-recipe)

(define (warmup-steps total)
  (exact-ceiling (/ total 4)))

(define-trainer FashionClassifier (net)
  (define-init () (set! net (fashion-net)))
  (define-forward (x) (net x))
  (define-optimizers
    (define opt (adamw (weight-decay-param-groups net #:weight-decay weight-decay) #:lr lr))
    (define total (run-estimated-stepping-batches run))
    (define warmup (warmup-steps total))
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

@bold{The fit.} One seeded @racket[fit] over a training and a test
@racket[tensor-dataset]: shuffled training batches from a seeded generator, the test
set after every epoch, two batches accumulated per step, the gradient norm clipped at
@racket[clip-val], at the precision and on the device given. The tests run it on the
committed fixture and the headline on the full sets. It answers the run and the
trained classifier.

@chunk[<r20-fit>
(define (train-fashion-mnist train-set test-set
                             #:batch-size batch-size #:test-batch-size test-batch-size
                             #:epochs epochs #:device device #:precision precision
                             #:log-every-n-steps [log-every 50] #:logger [logger #f]
                             #:callbacks [callbacks '()])
  (manual-seed! seed)
  (define classifier (FashionClassifier))
  (define run
    (fit classifier
         (dataloader train-set #:batch-size batch-size #:shuffle? #t
                     #:generator (make-generator seed))
         #:val (dataloader test-set #:batch-size test-batch-size)
         #:device device #:precision precision #:max-epochs epochs
         #:accumulate-grad-batches accumulate #:gradient-clip-val clip-val
         #:log-every-n-steps log-every #:logger logger #:callbacks callbacks))
  (values run classifier))]

@bold{The result.} @filepath{scripts/fashion-mnist-results.rkt} ran this program and
its twin at commit @tt{@commit} on @hardware on @run-date,
@(~a (length (hash-ref ours 'epochs))) epochs at precision @tt{@precision}, seed
@(~a run-seed).
Test accuracy after the last epoch is @(percent (hash-ref ours 'accuracy)) percent
here and @(percent (hash-ref theirs 'accuracy)) percent in Lightning
@lightning-version on PyTorch @|python-torch|.

@(tabular
  #:style 'boxed
  #:sep (hspace 2)
  (cons (list (bold "epoch") (bold "side") (bold "train loss") (bold "test loss")
              (bold "test accuracy") (bold "seconds"))
        (for*/list ([(our-row their-row) (in-parallel (hash-ref ours 'epochs)
                                                      (hash-ref theirs 'epochs))]
                    [(side row) (in-parallel '("Racket" "Lightning") (list our-row their-row))])
          (match-define (list epoch train-loss val-loss acc seconds) row)
          (list (~a epoch) side (real->decimal-string train-loss 4)
                (real->decimal-string val-loss 4) (string-append (percent acc) "%")
                (real->decimal-string seconds 1)))))

@chunk[<*>
  <r20-require>
  <r20-provide>
  <r20-model>
  <r20-trainer>
  <r20-fit>]
