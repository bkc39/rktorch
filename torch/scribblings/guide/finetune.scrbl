#lang scribble/manual
@(require (only-in racket/format ~a ~r)
          (only-in racket/list last)
          "../common.rkt"
          (for-label (except-in racket/base
                                abs cos exp log sin sort sqrt max min length
                                + - * /)
                     torch
                     torch/nn
                     torch/data/loader
                     torch/vision/hymenoptera
                     torch/vision/image-folder
                     torch/vision/resnet
                     torch/vision/transforms))

@(define results
   (call-with-input-file
     (collection-file-path "finetune.rktd" "torch" "scribblings" "results")
     read))

@(define (percent x) (~r (* 100 x) #:precision '(= 1)))

@(define (best phase)
   (for/fold ([best 0]) ([row (in-list (hash-ref results 'epochs))]
                         #:when (eq? (car row) phase))
     (max best (list-ref row 3))))

@title[#:tag "finetune"]{Fine-tuning}

A pretrained network knows ImageNet's thousand classes. Most problems
have other classes and far fewer examples, and the usual answer is to
keep what the network has learned about images and teach it only the new
decision. This chapter does that for the dataset of PyTorch's
transfer-learning tutorial: 244 photographs of ants and bees, which would
be hopeless for training a ResNet from scratch and are plenty for
adapting one.

@section[#:tag "finetune-data"]{A folder per class}

The photographs come as one directory per class. An
@racket[image-folder] lists them once and decodes each on demand,
labelling the classes in the order of their names. The four photographs
committed with the library are laid out the same way:

@torch-examples[
(require torch/data/loader torch/vision/image-folder)
(define folder
  (collection-path "torch" "vision" "fixtures" "hymenoptera"))
(image-folder-classes folder)
(define photos (image-folder folder))
(length photos)
(define-values (image label) (dataset-ref photos 3))
(list (shape image) (item label))
]

@racket[hymenoptera-dataset] is the full set, fetched and cached the
first time it is asked for.

@section[#:tag "finetune-head"]{A new head}

The network's last layer maps 512 features to a thousand ImageNet
logits. For two classes it needs a fresh layer mapping them to two.
@racket[#:classes] builds the network that way: the backbone gets the
pretrained weights and the head starts from a fresh initialisation.

@torch-examples[
(require torch/vision/resnet)
(define net (resnet18 #:classes 2))
(shape (cdr (assoc "fc.weight" (named-parameters net))))
]

Here @racket[#:pretrained?] is left out, so the example builds without
fetching anything; the program below passes @racket[#t].

@section[#:tag "finetune-freeze"]{Freezing}

A weight is frozen by no longer requiring a gradient. The backward pass
stops short of it, and an optimizer given only the other weights never
moves it. Freezing everything but the head turns the network into a
fixed feature extractor with a small classifier on top:

@torch-examples[
(for ([(name p) (in-named-parameters net)])
  (requires-grad! p (regexp-match? #rx"^fc[.]" name)))
(for/list ([(name p) (in-named-parameters net)]
           #:when (requires-grad? p))
  name)
]

@section[#:tag "finetune-run"]{Two phases}

@filepath{examples/racket/15-finetune.rkt} trains in two phases. First
only the head learns, with the backbone frozen, at a rate of 0.001 with
momentum 0.9 and a tenfold decay every seven epochs. Then every weight is
unfrozen and the whole network trains at a tenth of that rate, so the
pretrained features move only as far as the new task needs. Training
draws a fresh augmentation of every photograph each epoch with
@racket[random-resized-crop] and @racket[random-horizontal-flip];
validation uses the fixed ImageNet preprocessing. Throughout, the network
stays in @racket['train] mode while it learns, so even the frozen batch
norms keep updating their running statistics to the new photographs.

This is a run of the program on @(hash-ref results 'device) on
@(hash-ref results 'date), @(~a (hash-ref results 'seconds)) seconds for
both phases. The head alone reaches
@(percent (best 'feature-extract)) percent validation accuracy at best,
and fine-tuning everything @(percent (best 'fine-tune)) percent, ending
at @(percent (list-ref (last (hash-ref results 'epochs)) 3)) percent; the
tutorial reports about 95. With 153 validation photographs, one
photograph is 0.65 points, so the wobble from epoch to epoch is a few
photographs changing sides.

@(tabular
  #:style 'boxed
  #:sep (hspace 2)
  (cons (list (bold "phase") (bold "epoch") (bold "train loss")
              (bold "val accuracy") (bold "seconds"))
        (for/list ([row (in-list (hash-ref results 'epochs))])
          (list (~a (list-ref row 0))
                (~a (list-ref row 1))
                (~r (list-ref row 2) #:precision '(= 4))
                (string-append (percent (list-ref row 3)) "%")
                (~r (list-ref row 4) #:precision '(= 1))))))

At fixture scale, the same two phases agree with PyTorch's: from the same
head, three steps of each on the four committed photographs give losses
within @tt{1e-4} of torchvision's, and so do the weights they leave
behind.
