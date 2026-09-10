#lang scribble/manual

@(require (for-label racket/base
                     (only-in torch T |@| tensor tensor? transpose ~>)))

@title{Transpose shorthand}

@defmodule[torch]

@defproc[(T [x tensor?]) tensor?]{
Returns a view with all dimensions in reverse order, matching Python's
@tt{x.T}. A matrix of shape @tt{[M, N]} becomes @tt{[N, M]}; a tensor of
shape @tt{[B, H, S, D]} becomes @tt{[D, S, H, B]}.
Scalars and vectors retain their shapes. Storage is shared with @racket[x],
and gradients propagate through the view.

@racketblock[
(T (tensor '((1 2 3) (4 5 6))))
(~> x T)
(|@| x (T weight))
]

For batched attention keys, use @racket[(transpose keys -2 -1)] to swap
only the last two dimensions. That operation corresponds to Python's
@tt{keys.mT}, whereas @racket[T] reverses every axis.

PyTorch deprecates @tt{.T} for tensors whose rank is not two.
This shorthand supports all ranks without emitting a deprecation warning.
}
