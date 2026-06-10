#lang scribble/lp2

@(require (for-label (except-in racket/base exp log sqrt max min + - * /)
                     torch))

@section[#:tag "ex-mlp"]{Training an MLP end to end}

The v1 capstone: a two-layer perceptron declared with @racket[define-module],
trained for a few SGD steps on a fixed random batch. The module is a plain
struct tree — @racket[parameters] recursively collects the four tensors the
optimizer updates, and the whole model is reclaimed by the garbage collector
when dropped.

@chunk[<r04-require>
(require torch
         torch/nn)]

@chunk[<r04-provide>
(provide run-example)]

@bold{The model.} Submodules are constructed in the @racket[#:submodules]
clause and are callable in @racket[#:forward] like Python's
@tt{self.fc1(x)} — @racket[prop:procedure] plays the role of
@tt{__call__}.

@chunk[<r04-model>
(define-module mlp (d-in d-hidden d-out)
  #:submodules ([fc1 (linear d-in d-hidden)]
                [fc2 (linear d-hidden d-out)])
  #:forward (x)
  (fc2 (relu (fc1 x))))]

@bold{The loop.} Each step: clear gradients, forward, MSE loss, backward,
update. With the same seed, this matches @filepath{python/04_mlp.py}
draw-for-draw: the two @racket[linear] inits consume the RNG exactly like
@tt{nn.Linear}, then the batch is sampled identically.

@chunk[<r04-run>
(define (run-example)
  (manual-seed! 0)
  (define net (mlp 4 8 2))
  (define x (randn 16 4))
  (define y (randn 16 2))
  (define opt (sgd (parameters net) #:lr 0.1))
  (define losses
    (for/list ([_ (in-range 5)])
      (zero-grads! opt)
      (define loss (mse-loss (net x) y))
      (backward! loss)
      (step! opt)
      (item loss)))
  (values losses net))]

@chunk[<*>
  <r04-require>
  <r04-provide>
  <r04-model>
  <r04-run>]
