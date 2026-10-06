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

@bold{AI disclosure.} Most of rktorch was written by AI coding agents:
Anthropic's Claude, working through Claude Code under the author's
direction. That covers the code, the tests and this manual, and most
commits carry a Claude co-author line. The author sets the design, reviews
the work and decides what is merged. The agents' work is held to the same
checks as everything else: the test suite, and parity tests that compare
the library's results with PyTorch's own on the same inputs.

@bold{License.} rktorch's code is under the Apache License 2.0. The data
fixtures it ships keep their sources' licenses, recorded in the
@filepath{NOTICE} beside each: the LibriSpeech utterance is CC BY 4.0 and
the Tatoeba sentence pairs CC BY 2.0 FR, so the package as a whole is
@tt{Apache-2.0 AND CC-BY-4.0 AND CC-BY-2.0-FR}. The photographs and the
@italic{Heart of Darkness} excerpt are in the public domain. libtorch
itself is under PyTorch's BSD-3-Clause license.

@bold{Acknowledgements.} The tensors, kernels and autograd are libtorch's,
the work of the PyTorch team. The design follows
@hyperlink["https://github.com/janestreet/torch"]{ocaml-torch}, by Laurent
Mazare and Jane Street, which this project keeps as a reference
implementation. Several worked examples follow published work: PyTorch's
tutorials on translation, transfer learning and style transfer, the neural
style algorithm of Gatys, Ecker and Bethge, and Karpathy's char-rnn. The
pretrained weights are torchvision's. The data comes from LibriSpeech
(Panayotov, Chen, Povey and Khudanpur), the Tatoeba Project, MNIST,
CIFAR-10 and Project Gutenberg. The photographs come from Wikimedia
Commons, among them one by Fredrik Lähnn, who asks to be credited.

@local-table-of-contents[]

@include-section["guide.scrbl"]
@include-section["reference.scrbl"]

@section{Status}

rktorch is a work in progress. The surface documented here is stable enough
to build on; the corners the manual does not reach yet are tracked in the
@hyperlink["https://github.com/bkc39/rktorch/issues"]{project's issues}.
