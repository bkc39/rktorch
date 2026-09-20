#lang scribble/manual

@(require "common.rkt"
          (for-label racket/base
                     racket/contract
                     (only-in torch relu shape tensor? zeros)
                     (only-in torch/nn
                              Conv2d Dropout Linear Sequential adam
                              define-layer layer? mse-loss named-parameters
                              parameters sgd step! zero-grads!)))

@title{Built-in layers, optimizers and losses}

@defmodule[torch/nn #:link-target? #f]

The concrete layers the library ships, and the machinery that trains them.
@racket[define-layer], the interface they implement, and the container
forms are described in @secref["Layers"].

@section{Layer constructors}

Every layer is applied like a procedure, and is @racket[layer?].
@racket[Sequential], which composes them, is described with the other
container forms in @secref["Layers"].

@defproc[(Linear [in exact-positive-integer?] [out exact-positive-integer?])
         layer?]{
The affine map @tt{xW@superscript{T} + b}, from @racket[in] features to
@racket[out]. Owns @racket["weight"] and @racket["bias"], initialized as
PyTorch's @tt{nn.Linear} initializes them.

@torch-examples[
(define fc (Linear 3 2))
(shape (fc (zeros 1 3)))
(map car (named-parameters fc))
]}

@defproc[(Conv2d [in exact-positive-integer?]
                 [out exact-positive-integer?]
                 [kernel exact-positive-integer?]
                 [#:stride stride exact-positive-integer? 1]
                 [#:padding padding exact-nonnegative-integer? 0])
         layer?]{
Two-dimensional convolution over a batch shaped @tt{[N, in, H, W]},
answering @tt{[N, out, H', W']} with the usual convolution arithmetic.

@torch-examples[
(shape ((Conv2d 1 4 3) (zeros 1 1 8 8)))
]}

@defproc[(Dropout [#:p p (and/c real? (between/c 0 1)) 0.5]) layer?]{
Zeroes each element independently with probability @racket[p] while the
layer is in training mode, and is the identity while it is evaluating. See
@secref["Layers"] for the mode flag.}

@section{Collecting parameters}

@defproc[(parameters [m layer?]) (listof tensor?)]{
Every parameter in the layer tree, depth first, each appearing once.}

@defproc[(named-parameters [m layer?] [prefix string? ""])
         (listof (cons/c string? tensor?))]{
As @racket[parameters], each paired with its dotted path through the tree
--- @racket["fc1.weight"] --- which is the name it carries in a
checkpoint.

@racket[prefix] is prepended verbatim, so a caller supplying one includes
its own separator: @racket["enc."] gives @racket["enc.fc1.weight"].

@torch-examples[
(map car (named-parameters (Sequential (Linear 2 2))))
]}

@defproc[(layer? [v any/c]) boolean?]{
Whether @racket[v] implements the layer interface.}

@section{Optimizers}

An optimizer holds the parameters it is responsible for and the rule for
updating them.

@margin-note{The contract on these is @tt{optimizer?}, a predicate defined
in @tt{torch/nn/optim} and not re-exported from @racketmodname[torch/nn];
it is written @racket[any/c] here because the name cannot be reached
through the usual import.}

@defproc[(sgd [params (listof tensor?)] [#:lr lr real?]) any/c]{
Stochastic gradient descent: each parameter moves against its gradient by
@racket[lr]. Answers the optimizer that @racket[step!] and
@racket[zero-grads!] take.}

@defproc[(adam [params (listof tensor?)]
               [#:lr lr real? 0.001]
               [#:beta1 beta1 real? 0.9]
               [#:beta2 beta2 real? 0.999]
               [#:eps eps real? 1e-8])
         any/c]{
Adam, with PyTorch's defaults.}

@defproc[(step! [opt any/c]) void?]{
Applies one update to every parameter the optimizer holds, from the
gradients currently accumulated.}

@defproc[(zero-grads! [opt any/c]) void?]{
Clears the gradient of every parameter the optimizer holds. Note that it
takes the @emph{optimizer}, not the parameter list. Because gradients
accumulate, a training loop that omits this descends using the sum of every
gradient computed so far.}

@section{Losses}

@defproc[(mse-loss [input tensor?] [target tensor?]) tensor?]{
Mean squared error between two tensors of the same shape, as a one-element
tensor.}
