#lang scribble/manual

@title[#:tag "guide"]{Guide}

A narrative tour of the library, in the order you meet it: a tensor, the
gradients it tracks, a layer, and the loop that puts the three together.
It leaves the exhaustive contracts to the @secref["reference"], and points
there as it goes.

It assumes you can read Racket and have met arrays or tensors somewhere
before, in PyTorch, NumPy or another array language; it does not assume you
know PyTorch's API.

@include-section["guide/welcome.scrbl"]
@include-section["guide/tensors.scrbl"]
@include-section["guide/autograd.scrbl"]
@include-section["guide/layers.scrbl"]
@include-section["guide/training.scrbl"]
