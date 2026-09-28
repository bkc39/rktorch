#lang scribble/manual
@(require "../common.rkt"
          (for-label (except-in racket/base
                                abs cos exp log sin sort sqrt max min length
                                + - * /)
                     torch
                     torch/nn))

@title[#:tag "autograd"]{Automatic differentiation}

Training a model means nudging its numbers in the direction that reduces a
loss. Finding that direction by hand is the part nobody wants to do, so the
library does it: mark the tensors you want derivatives for, compute
something from them, and ask for the derivative of the result.

@section[#:tag "autograd-tape"]{Marking a tensor}

@racket[requires-grad!] marks a tensor as one to track. From then on, every
operation involving it records what it did:

@torch-examples[
(require torch)
(define x (requires-grad! (tensor '(2.0 3.0))))
(requires-grad? x)
]

@section[#:tag "autograd-backward"]{Asking for the gradient}

@racket[backward!] walks that record backwards from a single-element
tensor, accumulating the derivative of it with respect to every tracked
tensor that fed into it. @racket[grad] reads the result:

@torch-examples[
(define y (sum (mul x x)))
(backward! y)
(grad x)
]

The function here is @tt{y = x₁² + x₂²}, whose derivative is @tt{2x}. At
@tt{x = (2, 3)} that is @tt{(4, 6)}, which is what came back.

@margin-note{Gradients @emph{accumulate}: calling @racket[backward!] again
adds to what @racket[grad] already holds rather than replacing it. This is
why a training loop clears them every step --- see @secref["training"].}

@section[#:tag "autograd-no-grad"]{Turning it off}

Recording costs time and memory, and you do not want it when you are merely
evaluating a model. @racket[with-no-grad] turns tracking off for the
duration of its body:

@torch-examples[
(grad-enabled?)
(with-no-grad (grad-enabled?))
]

Use it around evaluation, around metric computation, and around the manual
parameter updates inside an optimizer --- anywhere the result is not
something you intend to differentiate.

@racket[detach] does the same job for a single tensor, answering one that
shares storage but carries no history.
