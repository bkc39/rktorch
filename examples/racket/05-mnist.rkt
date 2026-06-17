#lang scribble/lp2

@(require (for-label (except-in racket/base exp log sqrt max min + - * /)
                     ;; keep the conv2d *layer* (torch/nn); drop the functional
                     ;; torch conv2d so the doc binding agrees with the body.
                     (except-in torch conv2d)
                     (except-in torch/nn max-pool2d flatten)))

@section[#:tag "ex-mnist"]{Training a convnet on MNIST}

The v2 capstone: a LeNet-ish convolutional network --- two
@racket[conv2d]/@racket[max-pool2d] stages into two @racket[linear] layers ---
trained on the handwritten-digit dataset with the @racket[adam] optimizer. The
same code path runs on the GPU when one is present and on the CPU otherwise; on
an RTX 3090 Ti it reaches ~98% test accuracy in three epochs (~1.2s each), and
~97% in a single CPU epoch.

Two surfaces share the name @racket[conv2d] (the functional op in @racketmodname[torch]
and the layer in @racketmodname[torch/nn]); @racket[max-pool2d] and @racket[flatten]
likewise. We keep the @racket[conv2d] @emph{layer} but the @racket[max-pool2d] /
@racket[flatten] @emph{functions}, so the @racket[except-in] forms below drop the
collisions in each direction. (Folding the F.* / nn.* split into one namespace is
issue #11; this is the workaround until then.)

@chunk[<r05-require>
(require (except-in torch conv2d)
         (except-in torch/nn max-pool2d flatten)
         (only-in torch/data/mnist load-mnist load-mnist-fixture))]

@chunk[<r05-provide>
(provide convnet pick-device run-example train-mnist)]

@bold{The model.} @racket[define-module] builds the parameter tree; the
submodules are callable in @racket[#:forward] exactly like @tt{self.c1(x)} in
PyTorch. The spatial arithmetic is the usual @tt{valid}-convolution bookkeeping:
@tt{28 -c3-> 26 -pool-> 13 -c3-> 11 -pool-> 5}, so the flattened feature map is
@tt{32*5*5 = 800} wide going into the first dense layer.

@chunk[<r05-model>
(define-module convnet ()
  #:submodules ([c1 (conv2d 1 16 3)]
                [c2 (conv2d 16 32 3)]
                [f1 (linear 800 128)]
                [f2 (linear 128 10)])
  #:forward (x)
  (let* ([h (max-pool2d (relu (c1 x)) 2)]
         [h (max-pool2d (relu (c2 h)) 2)]
         [h (relu (f1 (flatten h 1)))])
    (f2 h)))]

@bold{The device.} Pick the accelerator the way PyTorch does
(@tt{device = "cuda" if torch.cuda.is_available() else "cpu"}); setting it as
the process default means every tensor built afterwards --- the model's
parameters and each batch alike --- lands there, so the whole loop follows.

@chunk[<r05-device>
(define (pick-device)
  (if (cuda-available?) 'cuda 'cpu))]

@bold{Accuracy.} Evaluated in @racket[eval!] mode under @racket[call-with-no-grad]
(no autograd graph, no dropout), batched so the test set never has to live on the
device all at once. @racket[begin0] restores @racket[train!] mode on the way out,
including if a batch raises.

@chunk[<r05-accuracy>
(define (accuracy net xs ys)
  (eval! net)
  (begin0
    (call-with-no-grad
     (lambda ()
       (define n (car (tensor-shape xs)))
       (define correct
         (for/sum ([start (in-range 0 n 1000)])
           (define len (min 1000 (- n start)))
           (define preds (argmax (net (narrow xs 0 start len)) 1))
           (item (sum (eq preds (narrow ys 0 start len))))))
       (exact->inexact (/ correct n))))
    (train! net)))]

@bold{The deterministic core.} @racket[run-example] is the seeded, offline entry
the test harness and the PyTorch parity twin both drive: it trains a fresh
@racket[convnet] for @racket[steps] full-batch @racket[adam] steps on the
committed 256-image fixture and returns the per-step losses, the trained net, and
the device. Full-batch (no shuffling, no minibatch indexing) keeps it trivially
reproducible across both languages --- with the same seed the @racket[conv2d] and
@racket[linear] inits draw value-for-value like @tt{nn.Conv2d}/@tt{nn.Linear},
then the identical updates track @tt{torch.optim.Adam}. As in the MLP example the
process default device is saved and restored with @racket[dynamic-wind] so calling
this neither leaks the GPU onto later tensors nor clobbers a caller's choice.

@chunk[<r05-run>
(define (run-example #:steps [steps 5] #:device [device (pick-device)])
  (define saved (default-device))
  (dynamic-wind
    (lambda () (set-default-device! device))
    (lambda ()
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
      (values losses net device))
    (lambda () (set-default-device! saved))))]

@bold{The real thing.} @racket[train-mnist] is the headline run: it downloads the
full dataset (cached under @envvar{RKTORCH_MNIST_DIR} or the system cache dir),
trains for @racket[epochs] minibatched epochs, and reports held-out test accuracy
after each. This is what reaches ~98%; @racket[run-example] above is its offline,
fixture-sized shadow for testing.

@chunk[<r05-train-mnist>
(define (train-mnist #:epochs [epochs 3] #:batch [batch 128]
                     #:device [device (pick-device)])
  (define saved (default-device))
  (dynamic-wind
    (lambda () (set-default-device! device))
    (lambda ()
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
        (accuracy net test-x test-y)))
    (lambda () (set-default-device! saved))))]

@chunk[<*>
  <r05-require>
  <r05-provide>
  <r05-model>
  <r05-device>
  <r05-accuracy>
  <r05-run>
  <r05-train-mnist>]
