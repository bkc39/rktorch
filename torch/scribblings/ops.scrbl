#lang scribble/manual

@(require "common.rkt"
          (for-label (except-in racket/base
                                abs cos exp log sin sort sqrt max min length
                                + - * /)
                     racket/contract
                     torch
                     (only-in torch/data/loader dataloader define-dataset)))

@title{Operations on tensors}

@defmodule[torch #:link-target? #f]

@section{Shape}

@defproc[(reshape [t tensor?] [dim exact-integer?] ...) tensor?]{
A view of @racket[t] with the given shape, which must have the same number
of elements. One dimension may be @racket[-1], and is solved for.

@torch-examples[
(reshape (arange 6) 2 3)
(shape (reshape (arange 6) -1 3))
]}

@defproc[(transpose [t tensor?] [dim1 exact-integer?] [dim2 exact-integer?])
         tensor?]{
A view with the two named axes exchanged.

@torch-examples[(shape (transpose (zeros 2 3) 0 1))]}

@defidform[t]{
A terse alias for @racket[transpose], taking the same three arguments. See
also @racket[T], which reverses every axis.}

@defproc[(T [x tensor?]) tensor?]{
A view with every dimension in reverse order, matching Python's @tt{x.T}. A
matrix of shape @tt{[M, N]} becomes @tt{[N, M]}; a tensor of shape
@tt{[B, H, S, D]} becomes @tt{[D, S, H, B]}. Scalars and vectors keep their
shapes. Storage is shared with @racket[x], and gradients propagate through
the view.

@torch-examples[
(T (tensor '((1 2 3) (4 5 6))))
]

For batched attention keys, @racket[(transpose keys -2 -1)] swaps only the
last two dimensions, as Python's @tt{keys.mT} does, where @racket[T]
reverses every axis. PyTorch deprecates @tt{.T} for tensors whose rank is
not two; @racket[T] supports every rank without a warning.}

@section{Arithmetic}

@defproc[(mul [a (or/c tensor? real?)]
              [b (if (tensor? a) (or/c tensor? real?) tensor?)])
         tensor?]{
Elementwise product. A real operand broadcasts across the tensor; at least
one of the two must be a tensor.

A scalar crosses the FFI boundary as a C double, so an integer tensor
combined with an integer scalar comes back @racket['float32] where PyTorch
would keep the integer dtype.

@torch-examples[(tensor->list (mul (tensor '(1.0 2.0)) 3.0))]}

@defproc[(matmul [a tensor?] [b tensor?]) tensor?]{
Matrix product, following PyTorch's @tt{torch.matmul} broadcasting rules.

@torch-examples[
(matmul (tensor '((1.0 2.0) (3.0 4.0))) (tensor '((1.0 0.0) (0.0 1.0))))
]}

@defidform[|@|]{
The operator spelling of @racket[matmul], like Python's @tt{a @"@" b}.
Scribble reserves bare @litchar["@"], so it appears here as
@racket[|@|]; in ordinary code it is written @litchar["@"].}

@section{Threading}

@deftogether[(@defidform[~>] @defidform[~>>]
              @defidform[lambda~>] @defidform[lambda~>>])]{
Re-exported from the @hyperlink["https://docs.racket-lang.org/threading/"]{
@tt{threading}} library, so they are in scope with @racketmodname[torch]
and need no separate import.

@torch-examples[(~> (arange 6) (reshape 2 3) sum item)]}

@section{Elementwise functions}

@defproc[(relu [t tensor?]) tensor?]{
The rectifier, @tt{max(0, x)} elementwise.

@torch-examples[(relu (tensor '(-1.0 0.0 2.0)))]}

@section{Reductions}

@defproc[(sum [t tensor?]) tensor?]{
Adds every element, answering a one-element tensor. Use @racket[item] to
get a Racket number back.

@torch-examples[(sum (tensor '((1.0 2.0) (3.0 4.0))))]}

@defproc[(mean [t tensor?]) tensor?]{
The arithmetic mean of every element, as a one-element tensor.

@torch-examples[(mean (tensor '(1.0 2.0 3.0)))]}

@margin-note{Neither takes an axis argument yet: both are whole-tensor
reductions, where PyTorch's @tt{sum} and @tt{mean} accept a @tt{dim}.}

@section{Length}

@defproc[(length [v sized?]) exact-nonnegative-integer?]{
Python's @tt{len}, shadowing @racketmodname[racket/base]'s @racket[length]
the way @racket[+] is shadowed: a list, vector, string or hash answers what
it always did, a tensor answers its first dimension, and a dataset or
loader answers its number of items or batches.

@torch-examples[
(length '(1 2 3))
(length (zeros 4 2))
]

A rank-zero tensor has no length, as @tt{len} of a 0-d tensor raises.}

@defproc[(sized? [v any/c]) boolean?]{
Recognises anything @racket[length] accepts.}

@defthing[gen:sized any/c]{
The generic behind @racket[length], with that one method. A structure
implements it with @racket[#:methods]; @racket[define-dataset] does so for
every dataset, which is what lets @racket[length] answer a
@racket[dataloader].}

@section{Shadowed names}

A few operations share a name with @racketmodname[racket/base]. Each is
generic: a tensor argument dispatches to libtorch, and anything else defers
to the binding @racketmodname[racket/base] provides, so requiring
@racketmodname[torch] never breaks numeric code that was already in the
module. @racket[length], above, is the same kind of generic.

@deftogether[(@defidform[abs] @defidform[cos] @defidform[exp]
              @defidform[log] @defidform[sin] @defidform[sqrt]
              @defidform[max] @defidform[min] @defidform[sort])]{
Generic over tensors and the values @racketmodname[racket/base] accepts.}

@deftogether[(@defidform[+] @defidform[-] @defidform[*] @defidform[/])]{
The arithmetic operators, provided as renames rather than contracted
wrappers so the numeric path costs nothing. Numeric operands take
@racketmodname[racket/base]'s path; a tensor operand on either side
dispatches to the tensor operation, and a chain folds left.

@torch-examples[
(+ 1 2)
(+ (tensor '(1 2 3)) (tensor '(10 20 30)))
(+ (tensor '(1 2 3)) 10)
]}
