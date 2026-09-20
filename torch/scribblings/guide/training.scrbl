#lang scribble/manual
@(require "../common.rkt"
          (for-label (except-in racket/base
                                abs cos exp log sin sort sqrt max min length
                                + - * /)
                     torch
                     torch/nn))

@title[#:tag "training"]{A training loop}

Everything so far assembles into one small loop. This chapter fits a model
to data end to end, with no dataset machinery in the way --- see
@secref["Datasets_and_loaders"] for those.

@section[#:tag "training-problem"]{Something to learn}

Make inputs, and targets that are a fixed linear function of them. The
model does not know that function; the loop has to find it:

@torch-examples[
(require torch torch/nn)
(manual-seed! 0)
(define xs (randn 64 4))
(define w-true (tensor '((2.0) (-3.0) (0.5) (1.0))))
(define ys (|@| xs w-true))
(shape xs)
(shape ys)
]

@section[#:tag "training-pieces"]{The model and the optimizer}

@torch-examples[
(define-layer mlp (fc1 fc2)
  #:init (d-in d-hidden d-out)
  (set! fc1 (Linear d-in d-hidden))
  (set! fc2 (Linear d-hidden d-out))
  #:forward (x)
  (~> x fc1 relu fc2))
(define model (mlp 4 16 1))
(define opt (sgd (parameters model) #:lr 0.05))
]

An @deftech{optimizer} holds the parameters it is responsible for and the
rule for updating them. @racket[sgd] is plain stochastic gradient descent;
@racket[adam] is the adaptive one you will usually reach for on a real
problem.

@section[#:tag "training-step"]{One step}

A step is always the same four moves: measure, clear, differentiate,
update.

@torch-examples[
(define (train-step!)
  (define loss (mse-loss (model xs) ys))
  (zero-grads! opt)
  (backward! loss)
  (step! opt)
  (item loss))
(train-step!)
]

@racket[zero-grads!] is the one that catches people out. Gradients
accumulate, so without clearing them each step you would descend using the
sum of every gradient computed so far. Note that it takes the
@emph{optimizer}, not the parameter list.

@section[#:tag "training-loop"]{The loop}

Run the step until the loss stops falling:

@torch-examples[
(for ([i (in-range 200)]) (train-step!))
(train-step!)
]

That is the whole idea. A real loop replaces the fixed batch with an
iteration over a @racket[dataloader], evaluates on held-out data every so
often, and switches the model between @racket[train!] and @racket[eval!]
around those evaluations --- but the four moves in the middle do not
change.

@section[#:tag "training-device"]{Running it on a GPU}

@racket[with-default-device] sets the device every tensor built inside it
is allocated on, the model's parameters and the batch alike, so the same
loop runs on an accelerator without being rewritten:

@racketblock[
(with-default-device (accelerator-if-available)
  (define model (mlp 4 16 1))
  (define opt (sgd (parameters model) #:lr 0.05))
  (code:comment "... the loop, unchanged ...")
  (void))
]

@racket[accelerator-if-available] answers CUDA on a Linux machine with an
NVIDIA GPU, Metal on Apple Silicon, and the CPU otherwise.
