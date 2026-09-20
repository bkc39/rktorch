#lang scribble/lp2

@(require (for-label (except-in racket/base abs cos exp log sin sort sqrt max min length + - * /)
                     torch torch/nn torch/data/loader torch/data/mnist
                     torch/vision/ppm))

@section[#:tag "ex-vae"]{Training a variational autoencoder on MNIST}

The smallest end-to-end reparameterization: an encoder maps an image to
the mean and log-variance of a Gaussian over a 20-dimensional latent, a
draw from that Gaussian is decoded back to pixel logits, and the loss is
the reconstruction cross-entropy plus the KL divergence of the latent
distribution from the standard normal. Both halves are plain
@racket[Linear] layers with a ReLU between, the reference architecture of
the PyTorch examples repository.

@chunk[<r11-require>
(require torch torch/nn torch/data/loader
         (only-in torch/data/mnist load-mnist load-mnist-fixture)
         torch/vision/ppm)]

@chunk[<r11-provide>
(provide pick-device vae decode vae-loss run-example train-vae)]

@bold{The network.} The forward takes the image batch and the noise for
the reparameterization, so the draw is the caller's and a seeded run
replays it, and returns three values: the pixel logits, the means and the
log-variances. @racket[decode] alone maps latents to logits, for sampling.

@chunk[<r11-model>
(define-layer vae (enc mu-head logvar-head dec1 dec2)
  #:init (#:latent [latent 20] #:hidden [hidden 400])
  (set! enc (Linear 784 hidden))
  (set! mu-head (Linear hidden latent))
  (set! logvar-head (Linear hidden latent))
  (set! dec1 (Linear latent hidden))
  (set! dec2 (Linear hidden 784))
  #:forward (x eps)
  (define h (relu (enc (flatten x 1))))
  (define mu (mu-head h))
  (define logvar (logvar-head h))
  (define z (add mu (mul eps (exp (mul logvar 0.5)))))
  (values (dec2 (relu (dec1 z))) mu logvar))

(define (decode net z)
  ((child-ref net "dec2") (relu ((child-ref net "dec1") z))))]

@bold{The loss.} The binary cross-entropy of the logits against the pixels,
summed over the 784 pixels and averaged over the batch, plus the KL term
in closed form, summed over the latent and averaged over the batch: the
reference's sums divided by the batch size, so the number does not scale
with the batch.

@chunk[<r11-loss>
(define (pick-device)
  (accelerator-if-available))

(define (vae-loss logits x mu logvar)
  (define n (length x))
  (define recon
    (mul (binary-cross-entropy-with-logits logits (flatten x 1)) 784.0))
  (define kl
    (mul (sum (sub (sub (add 1.0 logvar) (mul mu mu)) (exp logvar)))
         (/ -0.5 n)))
  (add recon kl))]

@bold{The deterministic core.} @racket[run-example] builds the network under
one seed, reseeds so the noise draws replay on every device, and runs five
full-batch Adam steps on the committed 256-image fixture, returning the
per-step losses, the network and the device.

@chunk[<r11-run>
(define (run-example #:steps [steps 5] #:batch [batch #f]
                     #:device [device (pick-device)])
  (with-default-device device
    (manual-seed! 0)
    (define-values (all-xs _ys) (load-mnist-fixture))
    (define xs (if batch (narrow all-xs 0 0 batch) all-xs))
    (define net (vae))
    (define opt (adam (parameters net) #:lr 1e-3))
    (manual-seed! 0)
    (define losses
      (for/list ([_ (in-range steps)])
        (define eps (to (randn (length xs) 20 #:device 'cpu) device))
        (zero-grads! opt)
        (define-values (logits mu logvar) (net xs eps))
        (define loss (vae-loss logits xs mu logvar))
        (backward! loss)
        (step! opt)
        (item loss)))
    (values losses net device)))]

@bold{The headline run.} Shuffled minibatches of the full training set, the
mean loss per epoch, and after each epoch a 10x10 grid decoded from one
fixed set of latents, passed through a sigmoid into @tt{[0, 1]} and written
as a PPM into @racket[out]. Ten epochs on any device give the soft but
legible digits a linear VAE is known for; the loss settles near 105.

@chunk[<r11-train>
(define (train-vae #:epochs [epochs 10] #:batch [batch 128]
                   #:out [out #f] #:device [device (pick-device)])
  (with-default-device device
    (manual-seed! 0)
    (define-values (train-x _train-y) (load-mnist 'train))
    (define loader
      (dataloader (tensor-dataset train-x)
                  #:batch-size batch #:shuffle? #t
                  #:generator (make-generator 0)))
    (define net (vae))
    (define opt (adam (parameters net) #:lr 1e-3))
    (define fixed-z (randn 100 20))
    (for/list ([epoch (in-range epochs)])
      (define-values (total count)
        (for/fold ([total 0.0] [count 0]) ([(xb) (in-dataloader loader)])
          (define eps (randn (length xb) 20))
          (zero-grads! opt)
          (define-values (logits mu logvar) (net xb eps))
          (define loss (vae-loss logits xb mu logvar))
          (backward! loss)
          (step! opt)
          (values (+ total (* (length xb) (item loss))) (+ count (length xb)))))
      (when out
        (define samples
          (with-no-grad (reshape (sigmoid (decode net fixed-z)) 100 1 28 28)))
        (write-ppm (build-path out (format "vae-epoch-~a.ppm" (add1 epoch)))
                   (image-grid samples #:columns 10)
                   #:range '(0 1)))
      (/ total count))))]

@chunk[<*>
  <r11-require>
  <r11-provide>
  <r11-model>
  <r11-loss>
  <r11-run>
  <r11-train>]
