#lang scribble/manual
@(require (only-in racket/format ~r)
          (only-in racket/list drop-right last)
          (only-in racket/string string-join)
          "../common.rkt"
          (for-label (except-in racket/base
                                abs cos exp log sin sort sqrt max min length
                                + - * /)
                     torch
                     torch/nn
                     torch/vision/image
                     torch/vision/imagenet
                     torch/vision/resnet
                     torch/vision/transforms
                     torch/vision/weights))

@(define results
   (call-with-input-file
     (collection-file-path "imagenet-top5.rktd" "torch" "scribblings" "results")
     read))

@(define (photo-path rel)
   (apply collection-file-path
          (append (list (last rel)) (list "torch" "vision" "fixtures" "hymenoptera")
                  (drop-right rel 1))))

@title[#:tag "pretrained"]{A pretrained network}

Every network so far started from random weights. Most real vision work
starts instead from a network someone has already trained on a large
dataset, and this chapter uses one: torchvision's ResNet-18, trained on
the thousand classes of ImageNet, loaded into a Racket network of the
same shape. It takes three steps to use: decode a photograph, turn it
into exactly the input the network was trained on, and read the output
back as class names.

@section[#:tag "pretrained-read"]{Reading a photograph}

@racket[read-image] decodes a JPEG or a PNG into a @racket['uint8]
tensor, channels first. The @racket['rgb] mode makes sure a grayscale
photograph still has three channels:

@torch-examples[
(require torch/vision/image torch/vision/transforms)
(define bee
  (read-image (collection-file-path "10870992_eebeeb3a12.jpg" "torch"
                                    "vision" "fixtures" "hymenoptera" "bees")
              #:mode 'rgb))
(tensor-shape bee)
(tensor-dtype bee)
]

@section[#:tag "pretrained-input"]{The input the network expects}

The torchvision weights were trained and evaluated on 224 by 224 crops
whose pixels are floats in @tt{[0, 1]} standardised channel by channel
with the training set's mean and deviation. @racket[imagenet-preprocess]
is those steps in order: scale to floats, resize the shorter side to 256,
cut the central 224 by 224 window, and normalise.

@torch-examples[
(define x (imagenet-preprocess bee))
(tensor-shape x)
]

Feeding a network anything else, a different size or unnormalised
pixels, still produces an answer; it is just a worse one.

@section[#:tag "pretrained-load"]{Loading the weights}

@racket[resnet18] builds the network with torchvision's layout and field
names. With @racket[#:pretrained? #t] it fetches the published checkpoint
into the cache the first time, checks it, and loads it:

@racketblock[
(require torch/vision/resnet torch/vision/imagenet)
(define net (resnet18 #:pretrained? #t))
(define probs
  (in-eval-mode net
    (with-no-grad (softmax (net (unsqueeze x 0)) 1))))
(define-values (ps indices) (topk (select probs 0 0) 5))
(for/list ([p (in-list (tensor->list ps))]
           [i (in-list (tensor->list indices))])
  (cons (list-ref imagenet-classes i) p))
]

Two details matter. @racket[in-eval-mode] makes the batch norms use the
running statistics saved with the weights, so a batch of one image is
normalised the way ImageNet was; in training mode they would normalise
by the batch itself. And the checkpoint's keys are torchvision's, such as
@tt{layer1.0.bn1.running_mean}, while the Racket network's are
@tt{layer1.0.bn1.running-mean}: the builder loads with
@racket[load-state!]'s @racket[#:rename] and @racket[torchvision-key],
which rewrites one into the other. On the same pixels, the Racket
network's logits agree with torchvision's to within @tt{6e-5}.

@section[#:tag "pretrained-results"]{What it sees}

The four photographs committed with the library, two ants and two bees
from the ants-and-bees dataset, and ResNet-18's five most likely classes
for each, as @filepath{examples/racket/14-imagenet.rkt} reported them on
@(hash-ref results 'device) on @(hash-ref results 'date). The close-up of
an ant's underside, seen from an angle ImageNet rarely shows, is the one
the network gets wrong; "ant" is only its fourth guess.

@(apply
  itemlist
  (for/list ([entry (in-list (hash-ref results 'photos))])
    (item
     (image (photo-path (hash-ref entry 'path)) #:scale 0.25)
     (linebreak)
     (tt (string-join (hash-ref entry 'path) "/"))
     (tabular
      #:sep (hspace 2)
      (for/list ([guess (in-list (hash-ref entry 'top5))])
        (list (~r (* 100 (cdr guess)) #:precision '(= 1))
              (car guess)))))))

The same program runs on any JPEG:

@verbatim{racket examples/test/14-imagenet.rkt photo.jpg}
