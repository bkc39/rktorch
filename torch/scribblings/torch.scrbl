#lang scribble/manual

@title{The rktorch Reference}

Racket bindings for libtorch. This manual is the reference: it states what
each binding accepts and answers. For a narrative introduction --- a first
tensor through to a training loop --- start with
@other-doc['(lib "torch/scribblings/guide/guide.scrbl")].

@table-of-contents[]

@include-section["contract.scrbl"]
@include-section["shape.scrbl"]
@include-section["order.scrbl"]
@include-section["length.scrbl"]
@include-section["device.scrbl"]
@include-section["data.scrbl"]
@include-section["vision.scrbl"]
@include-section["nn.scrbl"]
@include-section["recurrent.scrbl"]
