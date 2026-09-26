#lang scribble/manual

@(require "common.rkt"
          (for-label racket/base
                     racket/contract
                     (only-in torch relu shape tensor? zeros)
                     (only-in torch/nn
                              Conv2d Dropout Linear Sequential
                              define-layer in-named-parameters layer? mse-loss
                              named-parameters parameters)))

@title{Built-in layers and losses}

@defmodule[torch/nn #:link-target? #f]

The concrete layers and losses the library ships. @racket[define-layer],
the interface the layers implement, and the container forms are described
in @secref["Layers"]; the optimizers and learning-rate schedules that
train them are in @secref["Optimizers and schedules"].

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
                 [kernel (or/c exact-positive-integer?
                               (list/c exact-positive-integer?
                                       exact-positive-integer?))]
                 [#:stride stride (or/c exact-positive-integer?
                                        (list/c exact-positive-integer?
                                                exact-positive-integer?))
                           1]
                 [#:padding padding (or/c exact-nonnegative-integer?
                                          (list/c exact-nonnegative-integer?
                                                  exact-nonnegative-integer?))
                            0])
         layer?]{
Two-dimensional convolution over a batch shaped @tt{[N, in, H, W]},
answering @tt{[N, out, H', W']} with the usual convolution arithmetic.

Each size is an integer for a square one, or a two-element list
@racket[(list height width)] for an asymmetric one, mirroring PyTorch's
@tt{nn.Conv2d(kernel_size=(3, 5))}.

@torch-examples[
(shape ((Conv2d 1 4 3) (zeros 1 1 8 8)))
]}

@defproc[(Dropout [#:p p (and/c real? (>=/c 0) (</c 1)) 0.5]) layer?]{
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

@defproc[(in-named-parameters [m layer?]) sequence?]{
@racket[named-parameters] as a sequence of two values per parameter, its
dotted path and its tensor, so a @racket[for] clause can bind both
without taking a pair apart; @tt{named_parameters()} in a Python
@tt{for} loop.

@torch-examples[
(for/list ([(name p) (in-named-parameters (Sequential (Linear 2 3)))])
  (list name (shape p)))
]}

@defproc[(layer? [v any/c]) boolean?]{
Whether @racket[v] implements the layer interface.}

@section{Losses}

@defproc[(mse-loss [input tensor?] [target tensor?]) tensor?]{
Mean squared error between two tensors of the same shape, as a one-element
tensor.}
