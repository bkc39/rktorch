#lang scribble/manual
@(require "../common.rkt")

@title{The rktorch Guide}

@author["bkc"]

This guide introduces @racketmodname[torch], Racket bindings for libtorch,
the library PyTorch is built on. It assumes you can read Racket and that
you have met arrays or tensors somewhere before, in PyTorch, NumPy or
another array language; it does not assume you know PyTorch's API.

The guide teaches the library in the order you meet it: a tensor, the
gradients the library tracks for you, a layer, and the loop that puts the
three together. It leaves the exhaustive contracts to
@other-doc['(lib "torch/scribblings/torch.scrbl")], and points there as it
goes.

Every example below is evaluated when this manual is built, so the results
are the ones the library actually produces.

@margin-note{rktorch is a work in progress. The surface documented here is
stable enough to build on; the corners the guide does not reach yet are
tracked in the repository's issues.}

@table-of-contents[]

@include-section["welcome.scrbl"]
@include-section["tensors.scrbl"]
@include-section["autograd.scrbl"]
@include-section["layers.scrbl"]
@include-section["training.scrbl"]
