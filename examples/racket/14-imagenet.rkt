#lang scribble/lp2

@(require (for-label (except-in racket/base abs cos exp log sin sort sqrt max min length + - * /)
                     torch torch/nn torch/vision/image torch/vision/imagenet
                     torch/vision/resnet torch/vision/transforms
                     torch/vision/weights))

@section[#:tag "ex-imagenet"]{Classifying photographs with a pretrained ResNet}

Every network in the examples so far starts from random weights. This one
starts from torchvision's: the ResNet of He, Zhang, Ren and Sun
(@italic{Deep Residual Learning for Image Recognition}, CVPR 2016), trained
on the thousand ImageNet classes and exported once, unchanged, to a
safetensors file this repository publishes. @racket[resnet18] builds the
network with torchvision's shape and field names and, with
@racket[#:pretrained? #t], fetches that file into the cache and loads it.
The program is the other half of using such a network: turning a JPEG into
exactly the input it was trained on, and reading its output back as class
names. It is ocaml-torch's @tt{pretrained} predict example.

@chunk[<r14-require>
(require torch torch/nn
         (only-in torch/vision/image read-image)
         (only-in torch/vision/imagenet imagenet-classes)
         (only-in torch/vision/resnet resnet18 resnet34 resnet50)
         (only-in torch/vision/transforms imagenet-preprocess))]

@chunk[<r14-provide>
(provide pick-device photos->batch classify top-k run-example)]

@bold{The input.} torchvision's ImageNet weights expect the preprocessing
they were evaluated with: the shorter side resized to 256 with the
antialiased bilinear filter, the central 224 by 224 window, the pixels as
floats in @tt{[0, 1]}, and each channel standardised with the training
set's mean and standard deviation. @racket[imagenet-preprocess] is those
four steps; the photographs are decoded as RGB so a grayscale JPEG still
has three channels.

@chunk[<r14-batch>
(define (pick-device)
  (accelerator-if-available))

(define (photos->batch paths #:device [device (pick-device)])
  (to (stack (for/list ([path (in-list paths)])
               (imagenet-preprocess (read-image path #:mode 'rgb)))
             0)
      device))]

@bold{The output.} The network answers a thousand logits per image. In
@racket['eval] mode its batch norms use the running statistics saved with
the weights rather than the batch's own, which is what makes one image a
meaningful batch; @racket[with-no-grad] skips building a graph nobody will
differentiate. A softmax turns the logits into probabilities, and
@racket[topk] picks the most likely classes.

@chunk[<r14-classify>
(define (classify net batch)
  (in-eval-mode net
    (with-no-grad
      (softmax (net batch) 1))))

(define (top-k probs [k 5])
  (for/list ([row (in-tensor probs)])
    (define-values (ps indices) (topk row k))
    (for/list ([p (in-flattened-tensor ps)]
               [index (in-flattened-tensor indices)])
      (cons (list-ref imagenet-classes index) p))))]

@bold{Putting it together.} The runner passes the committed ant and bee
photographs, or whatever paths it is given; @racket[#:pretrained? #f]
keeps the weights random, which is how the tests exercise the program
without the download.

@chunk[<r14-run>
(define (run-example paths
                     #:model [model resnet18]
                     #:pretrained? [pretrained? #t]
                     #:device [device (pick-device)])
  (define net (to (model #:pretrained? pretrained?) device))
  (top-k (classify net (photos->batch paths #:device device))))]

@chunk[<*>
<r14-require>
<r14-provide>
<r14-batch>
<r14-classify>
<r14-run>]
