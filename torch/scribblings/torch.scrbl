#lang scribble/manual

@(require "common.rkt"
          (for-label (except-in racket/base
                                abs cos exp log sin sort sqrt max min length
                                + - * /)
                     torch
                     torch/nn))

@title{rktorch: Racket bindings to libtorch}
@author{bkc}

@defmodule[torch]

@racketmodname[torch] binds
@hyperlink["https://pytorch.org/cppdocs/"]{libtorch}, the C++ library
underneath @hyperlink["https://pytorch.org/"]{PyTorch}. The tensors are the
same tensors: the same dtypes, the same broadcasting, the same autograd
engine, the same kernels on the same GPU. What changes is the language
around them.

The library is split across a few modules:

@itemlist[

 @item{@racketmodname[torch] --- tensors, the creation family, the
 elementwise and reduction operations, autograd, and devices.}

 @item{@racketmodname[torch/nn] --- @racket[define-layer] and the layer
 interface, the built-in layers, the optimizers and the losses.}

 @item{@racketmodname[torch/data/loader] --- datasets, batching and
 iteration.}

]

This manual has two parts. The @secref["guide"] is a narrative
introduction, from a first tensor through to a training loop; the
@secref["reference"] states what each binding accepts and answers.

Every example in this manual is evaluated when it is built, so the printed
results are the ones the library produces.

@local-table-of-contents[]

@include-section["guide.scrbl"]
@include-section["reference.scrbl"]

@section{Status}

rktorch is a work in progress. The surface documented here is stable enough
to build on; the corners the manual does not reach yet are tracked in the
@hyperlink["https://github.com/bkc39/rktorch/issues"]{project's issues}.
