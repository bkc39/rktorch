#lang scribble/manual

@(require "common.rkt"
          (for-label (except-in racket/base
                                abs cos exp log sin sort sqrt max min length
                                + - * /)
                     racket/contract
                     torch
                     (only-in torch/nn nll-loss)))

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

@defproc[(unsqueeze [t tensor?] [dim exact-integer?]) tensor?]{
A view of @racket[t] with a new dimension of size one inserted at
@racket[dim], which counts from the end when negative; @tt{torch.unsqueeze}.
It is how one image becomes a batch of one.

@torch-examples[(shape (unsqueeze (zeros 3 4) 0))]}

@defproc[(select [t tensor?] [dim exact-integer?] [index exact-integer?])
         tensor?]{
The slice of @racket[t] at @racket[index] along @racket[dim], a view with
that dimension removed; @tt{torch.select}, and @tt{t[i]} along the first
dimension. A negative @racket[index] counts from the end.

@torch-examples[(tensor->list (select (tensor '((1 2 3) (4 5 6))) 0 1))]}

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

@section{Softmax}

@defproc[(softmax [t tensor?] [dim exact-integer?]) tensor?]{
Exponentiates the entries of @racket[t] along @racket[dim] and divides
each by their sum, so every slice along @racket[dim] is a probability
distribution; @tt{torch.softmax}.

@torch-examples[(tensor->list (softmax (tensor '(1.0 2.0 3.0)) 0))]}

@defproc[(log-softmax [t tensor?] [dim exact-integer?]) tensor?]{
The logarithm of @racket[softmax], computed as each entry minus the
log-sum-exp along @racket[dim], so it stays finite where
@racket[softmax]'s entries underflow to zero; @tt{torch.log_softmax}. It
is what @racket[nll-loss] takes.}

@section{Shadowed names}

A few operations share a name with @racketmodname[racket/base]. Each is
generic: a tensor argument dispatches to libtorch, and anything else defers
to the binding @racketmodname[racket/base] provides, so requiring
@racketmodname[torch] never breaks numeric code that was already in the
module.

@deftogether[(@defidform[abs] @defidform[cos] @defidform[exp]
              @defidform[log] @defidform[sin] @defidform[sqrt]
              @defidform[max] @defidform[min] @defidform[sort]
              @defidform[length])]{
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
