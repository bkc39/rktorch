#lang scribble/lp2

@(require (only-in racket/format ~r)
          (for-label (except-in racket/base abs cos exp log sin sort sqrt max min length + - * /)
                     torch torch/nn torch/vision/image torch/vision/transforms
                     torch/vision/vgg))

@title[#:tag "ex-style-transfer"]{Painting a photograph in the style of another}

Neural style transfer, after Gatys, Ecker and Bethge (@italic{A Neural
Algorithm of Artistic Style}, 2015), PyTorch's tutorial and ocaml-torch's
@tt{neural_transfer} example. Nothing here is trained: the network is
VGG-16's convolutional half with torchvision's ImageNet weights, frozen,
and what the optimizer changes is the pixels of an image. The image starts
as the photograph and is pushed two ways at once. Its activations deep in
the network should stay close to the photograph's, which keeps what the
picture shows; and the correlations between feature channels, their Gram
matrices, should match the painting's at five depths, which carries over
brush strokes, colour and texture while ignoring where things are.

@chunk[<r16-require>
(require torch torch/nn
         (only-in torch/vision/image read-image)
         (only-in torch/vision/transforms
                  convert-image-dtype imagenet-normalize resize)
         (only-in torch/vision/vgg vgg16-features))]

@chunk[<r16-provide>
(provide pick-device load-image frozen-vgg style-layers content-layer
         activations gram-matrix style-transfer)]

@bold{The images.} Each is decoded to RGB, scaled to floats in
@tt{[0, 1]}, resized so its shorter side is @racket[size], and given a
batch dimension of one.

@chunk[<r16-data>
(define (pick-device)
  (accelerator-if-available))

(define (load-image path size device)
  (unsqueeze (resize (convert-image-dtype
                      (read-image path #:mode 'rgb #:device device))
                     size)
             0))]

@bold{The network.} VGG-16's features are thirteen 3-by-3 convolutions
with a ReLU after each and a 2-by-2 max-pool after every stage. Its weights
never change, so none of them requires a gradient; the gradient still
flows through them to the image.

@chunk[<r16-network>
(define (frozen-vgg device #:pretrained? [pretrained? #t])
  (define net (to (vgg16-features #:pretrained? pretrained?) device))
  (for ([p (in-list (parameters net))])
    (requires-grad! p #f))
  net)]

@bold{Where to look.} The tutorial's choice, which is also ocaml-torch's:
style at the first five convolutions, indices 0, 2, 5, 7 and 10 of the
features, and content at the fourth, index 7. A convolution's output is
read before its ReLU, as the tutorial reads it.

@chunk[<r16-layers>
(define style-layers '(0 2 5 7 10))
(define content-layer 7)]

@bold{Activations.} The image is normalised with ImageNet's statistics,
as the network was trained, and run through the features step by step,
keeping the outputs at the wanted indices and stopping after the last.

@chunk[<r16-activations>
(define (activations net image layers)
  (define last-layer (apply max layers))
  (for/fold ([x (imagenet-normalize image)]
             [found (hash)]
             #:result found)
            ([step (in-layers (child-ref net "features"))]
             [i (in-naturals)]
             #:break (> i last-layer))
    (define y (forward step x))
    (values y (if (memv i layers) (hash-set found i y) found))))]

@bold{Gram matrices.} A feature map of shape @tt{[N, C, H, W]} becomes a
@tt{C}-by-@tt{C} matrix of how strongly each pair of channels fires
together, summed over every position, so where a texture appears no
longer matters. Dividing by the number of entries keeps the five layers
on comparable scales.

@chunk[<r16-gram>
(define (gram-matrix features)
  (define-values (n c h w) (apply values (shape features)))
  (define m (reshape features (* n c) (* h w)))
  (div (matmul m (transpose m 0 1)) (* n c h w)))]

@bold{The optimisation.} The targets are computed once, without a
gradient. The image starts as a copy of the photograph, and every step
runs it through the network, sums the style losses at the five layers and
the content loss, weighs the style a million times more heavily, and takes
an Adam step on the pixels, which are then clamped back into
@tt{[0, 1]}. Each step reports its step number and both losses.

@chunk[<r16-transfer>
(define (style-transfer net content style
                        #:steps [steps 300]
                        #:lr [lr 0.02]
                        #:style-weight [style-weight 1e6]
                        #:content-weight [content-weight 1.0])
  (define-values (style-grams content-target)
    (with-no-grad
      (define s (activations net style style-layers))
      (values (for/hash ([i (in-list style-layers)])
                (values i (gram-matrix (hash-ref s i))))
              (hash-ref (activations net content (list content-layer))
                        content-layer))))
  (define image (zeros-like content))
  (copy! image content)
  (requires-grad! image #t)
  (define opt (adam (list image) #:lr lr))
  (define losses
    (for/list ([step (in-range 1 (add1 steps))])
      (zero-grads! opt)
      (define found
        (activations net image (cons content-layer style-layers)))
      (define style-loss
        (for/fold ([total 0.0])
                  ([i (in-list style-layers)])
          (add total (mse-loss (gram-matrix (hash-ref found i))
                               (hash-ref style-grams i)))))
      (define content-loss
        (mse-loss (hash-ref found content-layer) content-target))
      (backward! (add (mul style-loss style-weight)
                      (mul content-loss content-weight)))
      (step! opt)
      (with-no-grad
        (copy! image (clamp image #:min 0.0 #:max 1.0)))
      (list step (item style-loss) (item content-loss))))
  (values (detach image) losses))]

@section{What it produces}

@(define results
   (call-with-input-file
     (collection-file-path "style-transfer.rktd" "torch" "scribblings" "results")
     read))

@(define (fixture . path)
   (apply collection-file-path (car (reverse path))
          "torch" "vision" "fixtures" (reverse (cdr (reverse path)))))

The honey bee from the library's fixtures, @italic{The Starry Night}, and
what @(number->string (hash-ref results 'steps)) steps make of them at
@(number->string (hash-ref results 'size)) pixels, as
@filepath{scripts/style-transfer-results.rkt} ran them on
@(hash-ref results 'device) in @(number->string (hash-ref results 'seconds))
seconds on @(hash-ref results 'date):

@tabular[#:sep @hspace[1]
         (list (list (image (fixture "hymenoptera" "bees" "honey-bee.jpg")
                            #:scale 0.5)
                     (image (fixture "style" "starry-night.jpg") #:scale 0.5)
                     (image (collection-file-path "style-transfer.png" "torch"
                                                  "scribblings" "results")
                            #:scale 0.5)))]

The style loss falls by more than three orders of magnitude while the
content loss, zero while the image is still the photograph, climbs and
settles: the image gives up some of the photograph to take on the
painting.

@tabular[#:sep @hspace[2]
         (cons (list @bold{step} @bold{style loss} @bold{content loss})
               (for/list ([record (in-list (hash-ref results 'losses))]
                          #:when (or (= 1 (car record))
                                     (zero? (modulo (car record) 200))))
                 (list (number->string (car record))
                       (~r (cadr record) #:notation 'exponential #:precision 2)
                       (~r (caddr record) #:precision 1))))]

@chunk[<*>
<r16-require>
<r16-provide>
<r16-data>
<r16-network>
<r16-layers>
<r16-activations>
<r16-gram>
<r16-transfer>]
