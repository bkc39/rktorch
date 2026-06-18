#lang scribble/lp2

@(require (for-label (except-in racket/base exp log sqrt max min + - * /)
                     torch torch/nn))

@section[#:tag "ex-mnist"]{Training a convnet on MNIST}

The v2 capstone: a LeNet-ish convolutional network --- two
@racket[Conv2d]/@racket[max-pool2d] stages into two @racket[Linear] layers ---
trained on the handwritten-digit dataset with the @racket[adam] optimizer. The
same code path runs on the GPU when one is present and on the CPU otherwise; on
an RTX 3090 Ti it reaches ~98% test accuracy in three epochs (~1.2s each), and
~97% in a single CPU epoch.

The @emph{layer} constructors are PascalCase (@racket[Conv2d], @racket[Linear]),
mirroring PyTorch's @tt{torch.nn.Conv2d} classes; the @emph{functional} ops stay
lowercase on @racketmodname[torch] (@racket[max-pool2d], @racket[flatten],
@racket[relu], like @tt{torch.max_pool2d}). Because the two casings differ,
@racket[(require torch torch/nn)] never collides (#11) --- no prefix or
@racket[except-in] needed.

@chunk[<r05-require>
(require torch torch/nn
         (only-in torch/data/mnist load-mnist load-mnist-fixture))]

@chunk[<r05-provide>
(provide convnet pick-device accuracy run-example train-mnist)]

@bold{The model.} @racket[define-module] builds the parameter tree; the
submodules are callable in @racket[#:forward] exactly like @tt{self.c1(x)} in
PyTorch. The spatial arithmetic is the usual @tt{valid}-convolution bookkeeping:
@tt{28 -c3-> 26 -pool-> 13 -c3-> 11 -pool-> 5}, so the flattened feature map is
@tt{32*5*5 = 800} wide going into the first dense layer.

@chunk[<r05-model>
(define-module convnet ()
  #:submodules ([c1 (Conv2d 1 16 3)]
                [c2 (Conv2d 16 32 3)]
                [f1 (Linear 800 128)]
                [f2 (Linear 128 10)])
  #:forward (x)
  (~> x
      c1 relu (max-pool2d 2)
      c2 relu (max-pool2d 2)
      (flatten 1) f1 relu
      f2))]

@bold{The device.} Pick the accelerator the way PyTorch does
(@tt{device = "cuda" if torch.cuda.is_available() else "cpu"}); setting it as
the process default means every tensor built afterwards --- the model's
parameters and each batch alike --- lands there, so the whole loop follows.

@chunk[<r05-device>
(define (pick-device)
  (if (cuda-available?) 'cuda 'cpu))]

@bold{Accuracy.} Evaluated in @racket[eval!] mode under @racket[with-no-grad] (no
autograd graph, no dropout), batched so the test set never has to live on the
device all at once. @racket[in-eval-mode] flips to @racket[eval!] for the body
and restores the prior mode on the way out, @emph{even if a batch raises} (it is
@racket[dynamic-wind] underneath) — so calling @racket[accuracy] mid-training
can't leave the model stuck in eval mode.

@chunk[<r05-accuracy>
(define (accuracy net xs ys)
  (in-eval-mode net
    (with-no-grad
      (define n (car (tensor-shape xs)))
      (define correct
        (for/sum ([start (in-range 0 n 1000)])
          (define len (min 1000 (- n start)))
          (define preds (argmax (net (narrow xs 0 start len)) 1))
          (item (sum (eq preds (narrow ys 0 start len))))))
      (exact->inexact (/ correct n)))))]

@bold{The deterministic core.} @racket[run-example] is the seeded, offline entry
the test harness and the PyTorch parity twin both drive: it trains a fresh
@racket[convnet] for @racket[steps] full-batch @racket[adam] steps on the
committed 256-image fixture and returns the per-step losses, the trained net, and
the device. Full-batch (no shuffling, no minibatch indexing) keeps it trivially
reproducible across both languages --- with the same seed the @racket[Conv2d] and
@racket[Linear] inits draw value-for-value like @tt{nn.Conv2d}/@tt{nn.Linear},
then the identical updates track @tt{torch.optim.Adam}. As in the MLP example the
process default device is set for the dynamic extent of the run with
@racket[with-default-device] (the device analogue of @racket[with-no-grad]), so
calling this neither leaks the GPU onto later tensors nor clobbers a caller's
choice — the prior default is restored even if a step raises.

@chunk[<r05-run>
(define (run-example #:steps [steps 5] #:device [device (pick-device)])
  (with-default-device device
    (manual-seed! 0)
    (define-values (xs ys) (load-mnist-fixture))
    (define net (convnet))
    (define opt (adam (parameters net) #:lr 0.001))
    (define losses
      (for/list ([_ (in-range steps)])
        (zero-grads! opt)
        (define loss (cross-entropy (net xs) ys))
        (backward! loss)
        (step! opt)
        (item loss)))
    (values losses net device)))]

@bold{The real thing.} @racket[train-mnist] is the headline run: it downloads the
full dataset (cached under @envvar{RKTORCH_MNIST_DIR} or the system cache dir),
trains for @racket[epochs] minibatched epochs, and reports held-out test accuracy
after each. This is what reaches ~98%; @racket[run-example] above is its offline,
fixture-sized shadow for testing.

@chunk[<r05-train-mnist>
(define (train-mnist #:epochs [epochs 3] #:batch [batch 128]
                     #:device [device (pick-device)])
  (with-default-device device
    (manual-seed! 0)
    (define-values (train-x train-y) (load-mnist 'train))
    (define-values (test-x test-y) (load-mnist 'test))
    (define n-train (car (tensor-shape train-x)))
    (define net (convnet))
    (define opt (adam (parameters net) #:lr 0.001))
    (for/list ([epoch (in-range epochs)])
      (for ([start (in-range 0 n-train batch)])
        (define len (min batch (- n-train start)))
        (zero-grads! opt)
        (define loss (cross-entropy (net (narrow train-x 0 start len))
                                    (narrow train-y 0 start len)))
        (backward! loss)
        (step! opt))
      (accuracy net test-x test-y))))]

@chunk[<*>
  <r05-require>
  <r05-provide>
  <r05-model>
  <r05-device>
  <r05-accuracy>
  <r05-run>
  <r05-train-mnist>]
