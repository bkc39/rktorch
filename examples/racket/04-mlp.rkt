#lang scribble/lp2

@(require (for-label (except-in racket/base exp log sqrt max min + - * /)
                     torch torch/nn))

@section[#:tag "ex-mlp"]{Training an MLP end to end}

The v1 capstone: a two-layer perceptron declared with @racket[define-module],
trained for a few SGD steps on a fixed random batch. The module is a plain
struct tree — @racket[parameters] recursively collects the four tensors the
optimizer updates, and the whole model is reclaimed by the garbage collector
when dropped.

@chunk[<r04-require>
(require torch torch/nn)]

@chunk[<r04-provide>
(provide run-example)]

@bold{The model.} Submodules are constructed in the @racket[#:submodules]
clause and are callable in @racket[#:forward] like Python's
@tt{self.fc1(x)} — @racket[prop:procedure] plays the role of
@tt{__call__}.

@chunk[<r04-model>
(define-module mlp (d-in d-hidden d-out)
  #:submodules ([fc1 (Linear d-in d-hidden)]
                [fc2 (Linear d-hidden d-out)])
  #:forward (x)
  (fc2 (relu (fc1 x))))]

@bold{The device.} Pick the accelerator the way PyTorch does
(@tt{torch.accelerator.current_accelerator()}): set it as the
process default and every tensor built afterwards — the module's parameters and
the batch alike — is allocated there, so the whole loop runs on the GPU when one
is present and on the CPU otherwise. @racket[run-example] returns the device it
chose so callers can report it.

@chunk[<r04-device>
(define (pick-device)
  (accelerator-if-available))]

@bold{The loop.} Each step: clear gradients, forward, MSE loss, backward,
update. On the CPU path, with the same seed this matches
@filepath{python/04_mlp.py} draw-for-draw: the two @racket[Linear] inits consume
the RNG exactly like @tt{nn.Linear}, then the batch is sampled identically. (The
GPU uses its own RNG, so CUDA and MPS runs train just as well but draw
different values.)

@bold{Restoring the default.} @racket[set-default-device!] is a process-wide
side effect, so @racket[run-example] captures the prior default and wraps the
loop in @racket[dynamic-wind] to restore it on the way out (even if a step
raises) — calling it must neither leak the GPU onto later tensors nor clobber a
default the caller had already chosen. The device is also an optional argument
so callers can pin it.

@chunk[<r04-run>
(define (run-example [device (pick-device)])
  (define saved (default-device))
  (dynamic-wind
    (lambda () (set-default-device! device))
    (lambda ()
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
      (values losses net device))
    (lambda () (set-default-device! saved))))]

@chunk[<*>
  <r04-require>
  <r04-provide>
  <r04-model>
  <r04-device>
  <r04-run>]
