#lang scribble/lp2

@(require (for-label (except-in racket/base abs cos exp log sin sort sqrt max min length + - * /)
                     torch torch/nn torch/data/loader torch/vision/hymenoptera
                     torch/vision/image-folder torch/vision/resnet
                     torch/vision/transforms))

@section[#:tag "ex-finetune"]{Fine-tuning a pretrained ResNet on ants and bees}

Transfer learning, after PyTorch's tutorial (Sasank Chilamkurthy,
@italic{Transfer Learning for Computer Vision}) and ocaml-torch's
@tt{finetuning} example: 244 photographs of ants and bees are far too few
to train a ResNet from scratch, but plenty to adapt one that has already
learned ImageNet. The network is torchvision's ResNet-18 with its
thousand-way head replaced by a fresh two-way one, trained in two phases.
First it is a fixed feature extractor: every pretrained weight is frozen
and only the new head learns. Then every weight is unfrozen and the
whole network is fine-tuned at a tenth of the rate, so the pretrained
features move only as far as the new task needs. Both phases use SGD
with momentum under a step schedule.

@chunk[<r15-require>
(require torch torch/nn
         (only-in torch/data/loader dataset-ref)
         (only-in torch/vision/hymenoptera hymenoptera-dataset)
         (only-in torch/vision/resnet resnet18)
         (only-in torch/vision/transforms
                  convert-image-dtype imagenet-normalize imagenet-preprocess
                  random-horizontal-flip random-resized-crop))]

@chunk[<r15-provide>
(provide pick-device load-train load-val train-batch head? set-frozen!
         accuracy train-phase finetune)]

@bold{The data.} An image folder reads one class per directory, labels in
the directories' alphabetical order, so ants are 0 and bees 1. There are
few enough photographs to decode each once and keep it on the device.
Training draws a fresh augmentation of every photograph each epoch, so
it keeps them whole, as floats; validation preprocesses each once, the way
ImageNet evaluation does, into one batch.

@chunk[<r15-data>
(define (pick-device)
  (accelerator-if-available))

(define (items dataset)
  (for/lists (images labels) ([i (in-range (length dataset))])
    (dataset-ref dataset i)))

(define (load-train dataset device)
  (define-values (images labels) (items dataset))
  (values (for/vector ([image (in-list images)])
            (convert-image-dtype (to image device)))
          (to (stack labels 0) device)))

(define (load-val dataset device)
  (define-values (images labels) (items dataset))
  (values (stack (for/list ([image (in-list images)])
                   (imagenet-preprocess (to image device)))
                 0)
          (to (stack labels 0) device)))]

@bold{Augmentation.} The tutorial's: a random crop covering 8 to 100
percent of the photograph at an aspect ratio between 3:4 and 4:3, resized
to 224, then a coin-flip mirror. The draws come from a seeded generator,
so a run replays.

@chunk[<r15-batch>
(define (train-batch images labels indices generator)
  (define crops
    (for/list ([i (in-list indices)])
      (imagenet-normalize
       (random-resized-crop (vector-ref images i) 224 #:generator generator))))
  (values (random-horizontal-flip (stack crops 0) #:generator generator)
          (index-select labels 0 (tensor indices
                                         #:device (tensor-device labels)))))]

@bold{Freezing.} A frozen weight does not require a gradient: the
backward pass stops short of it and an optimizer never moves it. The head
is the network's @racket[fc] field, so its parameters are the ones named
under it.

@chunk[<r15-freeze>
(define (head? name) (regexp-match? #rx"^fc[.]" name))

(define (set-frozen! net frozen?)
  (for ([np (in-list (named-parameters net))]
        #:unless (head? (car np)))
    (requires-grad! (cdr np) (not frozen?))))]

@bold{Accuracy.} The validation batch under @racket[in-eval-mode], so
the batch norms normalise with their running statistics, in slices of 64.

@chunk[<r15-accuracy>
(define (accuracy net xs ys)
  (in-eval-mode net
    (with-no-grad
      (define n (car (tensor-shape xs)))
      (define correct
        (for/sum ([start (in-range 0 n 64)])
          (define len (min 64 (- n start)))
          (item (sum (eq (argmax (net (narrow xs 0 start len)) 1)
                         (narrow ys 0 start len))))))
      (exact->inexact (/ correct n)))))]

@bold{One phase.} Epochs of shuffled batches, the network in
@racket['train] mode throughout, so its batch norms normalise by the batch
and keep updating their running statistics even while their weights are
frozen, as the tutorial's do. The optimizer steps every batch and the
schedule once an epoch. Each epoch reports its mean training loss, the
validation accuracy after it, and its wall-clock seconds.

@chunk[<r15-phase>
(define (train-phase net opt schedule train val
                     #:epochs epochs #:batch batch #:generator generator
                     #:phase phase)
  (define-values (images labels) (values (car train) (cdr train)))
  (define n (vector-length images))
  (for/list ([epoch (in-range 1 (add1 epochs))])
    (define start (current-inexact-milliseconds))
    (define order (tensor->list (randperm n #:generator generator)))
    (define losses
      (for/list ([from (in-range 0 n batch)])
        (define indices
          (for/list ([i (in-list order)]
                     [k (in-naturals)]
                     #:when (<= from k (sub1 (+ from batch))))
            i))
        (define-values (xs ys) (train-batch images labels indices generator))
        (zero-grads! opt)
        (define loss (cross-entropy (net xs) ys))
        (backward! loss)
        (step! opt)
        (item loss)))
    (step! schedule)
    (list phase epoch
          (/ (apply + losses) (length losses))
          (accuracy net (car val) (cdr val))
          (/ (- (current-inexact-milliseconds) start) 1000.0))))]

@bold{Both phases.} The head alone at 0.001 with momentum 0.9, the rate
falling tenfold every seven epochs; then everything at a tenth of that
under the same schedule. @racket[#:classes 2] gives the network a fresh
two-way head, where torchvision would assign a new @tt{model.fc}.

@chunk[<r15-finetune>
(define (finetune #:train [train-set (hymenoptera-dataset 'train)]
                  #:val [val-set (hymenoptera-dataset 'val)]
                  #:pretrained? [pretrained? #t]
                  #:feature-epochs [feature-epochs 10]
                  #:finetune-epochs [finetune-epochs 15]
                  #:batch [batch 4]
                  #:seed [seed 0]
                  #:device [device (pick-device)])
  (manual-seed! seed)
  (define generator (make-generator seed))
  (define net (to (resnet18 #:pretrained? pretrained? #:classes 2) device))
  (define-values (train-images train-labels) (load-train train-set device))
  (define-values (val-xs val-ys) (load-val val-set device))
  (define (phase name params lr epochs)
    (define opt (sgd params #:lr lr #:momentum 0.9))
    (train-phase net opt (step-lr opt #:step-size 7 #:gamma 0.1)
                 (cons train-images train-labels) (cons val-xs val-ys)
                 #:epochs epochs #:batch batch #:generator generator
                 #:phase name))
  (set-frozen! net #t)
  (define head (for/list ([np (in-list (named-parameters net))]
                          #:when (head? (car np)))
                 (cdr np)))
  (define features (phase 'feature-extract head 0.001 feature-epochs))
  (set-frozen! net #f)
  (define tuned (phase 'fine-tune (parameters net) 0.0001 finetune-epochs))
  (values (append features tuned) net))]

@chunk[<*>
<r15-require>
<r15-provide>
<r15-data>
<r15-batch>
<r15-freeze>
<r15-accuracy>
<r15-phase>
<r15-finetune>]
