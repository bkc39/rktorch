#lang scribble/manual
@(require "../common.rkt"
          (for-label (except-in racket/base
                                abs cos exp log sin sort sqrt max min length
                                + - * /)
                     torch
                     torch/nn))

@title[#:tag "guide-tensors"]{Tensors}

A @deftech{tensor} is a rectangular array of numbers with a single element
type, living on a single device. Everything else in the library is built on
it.

@section[#:tag "tensors-building"]{Building them}

@racket[tensor] takes Racket data --- a number, a list, a list of lists, as
deep as you like --- and the nesting becomes the shape:

@torch-examples[
(require torch)
(tensor 3.0)
(tensor '(1.0 2.0 3.0))
(tensor '((1.0 2.0) (3.0 4.0)))
]

The creation family builds tensors from a shape instead of from data.
@racket[zeros] and @racket[ones] take the dimensions as arguments, and
@racket[arange] counts:

@torch-examples[
(zeros 2 3)
(ones 2)
(arange 6)
]

@racket[randn] draws from a standard normal. Seed the generator with
@racket[manual-seed!] when you want the same draw twice:

@torch-examples[
(manual-seed! 0)
(randn 2 2)
]

@section[#:tag "tensors-queries"]{Asking about them}

@racket[shape] answers a list of dimensions, @racket[dtype] the element
type, and @racket[device] where the storage lives --- Python's
@tt{t.shape}, @tt{t.dtype} and @tt{t.device}:

@torch-examples[
(define m (tensor '((1 2 3) (4 5 6))))
(shape m)
(dtype m)
(device m)
]

@racket[device] does double duty: given a tensor it reads the tensor's
device, and given a device name it builds one, as @racket[(device 'cuda 1)].

@section[#:tag "tensors-out"]{Getting data back out}

@racket[tensor->list] flattens to a Racket list, and @racket[item] takes a
single-element tensor down to a Racket number:

@torch-examples[
(tensor->list (tensor '((1 2) (3 4))))
(item (sum (tensor '((1 2) (3 4)))))
]

@section[#:tag "tensors-ops"]{Operating on them}

Shape operations rearrange without touching the data:

@torch-examples[
(reshape (arange 6) 2 3)
(transpose (reshape (arange 6) 2 3) 0 1)
]

Elementwise operations apply to every element, and broadcast a scalar
across the whole tensor:

@torch-examples[
(relu (tensor '(-1.0 0.0 2.0)))
(* (tensor '(1.0 2.0 3.0)) 2.0)
]

Reductions collapse a tensor to one value. @racket[sum] and @racket[mean]
reduce the whole thing:

@torch-examples[
(sum (tensor '((1.0 2.0) (3.0 4.0))))
(mean (tensor '(1.0 2.0 3.0)))
]

@margin-note{Reducing along a chosen axis is not on the surface yet ---
today @racket[sum] and @racket[mean] are whole-tensor reductions. Reshape
or use @racket[matmul] where PyTorch would take a @tt{dim} argument.}

@section[#:tag "tensors-matmul"]{Matrix multiplication}

@racket[matmul] is the general form, and @racket[|@|] is the operator
spelling, like Python's @tt{a @"@" b}. (Scribble reserves bare
@litchar["@"], so the operator appears as @racket[|@|] in this rendered
chunk; in ordinary code it is just @litchar["@"].)

@torch-examples[
(define a (tensor '((1.0 2.0) (3.0 4.0))))
(define b (tensor '((1.0 0.0) (0.0 1.0))))
(matmul a b)
(|@| a b)
]

@racket[t] is a terse alias for @racket[transpose], and unary @racket[T]
reverses every axis the way Python's @tt{x.T} does.

@section[#:tag "tensors-threading"]{Threading}

rktorch re-provides @racket[~>] from the @tt{threading} library, so it is
already in scope. It reads better than nesting when a value flows through
several operations in a row:

@torch-examples[
(~> (arange 6) (reshape 2 3) sum item)
]
