#lang scribble/lp2

@(require (for-label (except-in racket/base abs cos exp log sin sort sqrt max min length + - * /)
                     torch torch/nn torch/data/loader torch/data/mnist
                     torch/vision/ppm))

@title[#:tag "ex-dcgan"]{Training a DCGAN on MNIST}

The first generative example that is not a diffusion model: two networks
in a contest. A generator maps a latent draw to an image through
transposed convolutions; a discriminator scores images as real or fake
through strided convolutions; each trains against the other's current
weights with a binary cross-entropy on the discriminator's logits. The
architecture is the DCGAN paper's shrunk to 28x28: batch norm and ReLU in
the generator, batch norm and leaky ReLU in the discriminator, a hyperbolic
tangent on the output, so the images live in @tt{[-1, 1]}.

@chunk[<r10-require>
(require torch torch/nn torch/data/loader
         (only-in torch/data/mnist load-mnist load-mnist-fixture)
         torch/vision/ppm)]

@chunk[<r10-provide>
(provide pick-device generator discriminator train-step run-example
         train-dcgan)]

@bold{The two networks.} The generator lifts a 100-dimensional draw to a
128-channel 7x7 map through a linear layer, then doubles the resolution
twice with transposed convolutions of kernel 4, stride 2 and padding 1;
the discriminator halves it twice with the same kernel shape and reads a
logit off the 128x7x7 map. The generator's first batch norm is one
dimensional, over the flat features, as in the reference code.

@chunk[<r10-model>
(define-layer generator (fc bn0 up1 bn1 up2)
  #:init (#:latent [latent 100])
  (set! fc (Linear latent (* 128 7 7)))
  (set! bn0 (BatchNorm1d (* 128 7 7)))
  (set! up1 (ConvTranspose2d 128 64 4 #:stride 2 #:padding 1))
  (set! bn1 (BatchNorm2d 64))
  (set! up2 (ConvTranspose2d 64 1 4 #:stride 2 #:padding 1))
  #:forward (z)
  (~> z fc bn0 relu (reshape (length z) 128 7 7) up1 bn1 relu up2 tanh))

(define-layer discriminator (conv1 conv2 bn fc)
  #:init ()
  (set! conv1 (Conv2d 1 64 4 #:stride 2 #:padding 1))
  (set! conv2 (Conv2d 64 128 4 #:stride 2 #:padding 1))
  (set! bn (BatchNorm2d 128))
  (set! fc (Linear (* 128 7 7) 1))
  #:forward (x)
  (~> x conv1 (leaky-relu #:negative-slope 0.2)
      conv2 bn (leaky-relu #:negative-slope 0.2)
      (flatten 1) fc))]

@bold{One step.} The discriminator first: its loss is the cross-entropy of
the real batch against ones plus that of a generated batch, detached so
the generator does not learn from this half, against zeros. Then the
generator: the same discriminator scores the fresh batch again, this time
with the graph intact, against ones. The latent draw is made on the CPU
whatever device the networks train on, so a seeded run replays it on the
GPU as the twin draws it. The two loss targets are built on
@racket[device] like the latent, so a caller may step a model that is not
on the default device. Images arrive from the loader in @tt{[0, 1]} and
are rescaled to the generator's range.

@chunk[<r10-step>
(define (pick-device)
  (accelerator-if-available))

(define (train-step gen disc opt-g opt-d xs device #:latent [latent 100])
  (define n (length xs))
  (define real (sub (mul xs 2.0) 1.0))
  (define z (to (randn n latent #:device 'cpu) device))
  (define ones-target (ones n 1 #:device device))
  (define zeros-target (zeros n 1 #:device device))
  (define fake (gen z))
  (zero-grads! opt-d)
  (define d-loss
    (add (binary-cross-entropy-with-logits (disc real) ones-target)
         (binary-cross-entropy-with-logits (disc (detach fake)) zeros-target)))
  (backward! d-loss)
  (step! opt-d)
  (zero-grads! opt-g)
  (define g-loss (binary-cross-entropy-with-logits (disc fake) ones-target))
  (backward! g-loss)
  (step! opt-g)
  (values (item d-loss) (item g-loss)))]

@bold{The deterministic core.} @racket[run-example] builds both networks
under one seed, generator first, then reseeds so the latent draws replay
on every device, and runs three full-batch steps on the committed 256-image
fixture with the paper's Adam settings, a rate of 0.0002 and a first
moment of 0.5. It returns the discriminator and generator losses per step,
both networks and the device.

@chunk[<r10-run>
(define (run-example #:steps [steps 3] #:batch [batch #f]
                     #:device [device (pick-device)])
  (with-default-device device
    (manual-seed! 0)
    (define-values (all-xs _ys) (load-mnist-fixture))
    (define xs (if batch (narrow all-xs 0 0 batch) all-xs))
    (define gen (generator))
    (define disc (discriminator))
    (define opt-g (adam (parameters gen) #:lr 2e-4 #:beta1 0.5))
    (define opt-d (adam (parameters disc) #:lr 2e-4 #:beta1 0.5))
    (manual-seed! 0)
    (define-values (d-losses g-losses)
      (for/fold ([ds '()] [gs '()] #:result (values (reverse ds) (reverse gs)))
                ([_ (in-range steps)])
        (define-values (d g) (train-step gen disc opt-g opt-d xs device))
        (values (cons d ds) (cons g gs))))
    (values d-losses g-losses gen disc device)))]

@bold{The headline run.} Shuffled minibatches of the full training set for
a few epochs, the last partial batch dropped because the generator's batch
norm needs more than one image, the mean losses per epoch reported, and
after each epoch a
10x10 grid of images from one fixed latent draw written as a PPM into
@racket[out], so the same hundred latents can be watched sharpening from
epoch to epoch. On an RTX 3090 Ti an epoch takes seconds; five epochs
give recognisable digits.

@chunk[<r10-train>
(define (train-dcgan #:epochs [epochs 5] #:batch [batch 128]
                     #:out [out #f] #:device [device (pick-device)])
  (with-default-device device
    (manual-seed! 0)
    (define-values (train-x _train-y) (load-mnist 'train))
    (define loader
      (dataloader (tensor-dataset train-x)
                  #:batch-size batch #:shuffle? #t #:drop-last? #t
                  #:generator (make-generator 0)))
    (define gen (generator))
    (define disc (discriminator))
    (define opt-g (adam (parameters gen) #:lr 2e-4 #:beta1 0.5))
    (define opt-d (adam (parameters disc) #:lr 2e-4 #:beta1 0.5))
    (define fixed-z (randn 100 100))
    (for/list ([epoch (in-range epochs)])
      (define-values (d-total g-total count)
        (for/fold ([d-total 0.0] [g-total 0.0] [count 0])
                  ([(xb) (in-dataloader loader)])
          (define-values (d g) (train-step gen disc opt-g opt-d xb device))
          (values (+ d-total (* (length xb) d))
                  (+ g-total (* (length xb) g))
                  (+ count (length xb)))))
      (when out
        (define samples (in-eval-mode gen (with-no-grad (gen fixed-z))))
        (write-ppm (build-path out (format "dcgan-epoch-~a.ppm" (add1 epoch)))
                   (image-grid samples #:columns 10)
                   #:range '(-1 1)))
      (list (/ d-total count) (/ g-total count)))))]

@chunk[<*>
  <r10-require>
  <r10-provide>
  <r10-model>
  <r10-step>
  <r10-run>
  <r10-train>]
