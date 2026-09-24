#lang scribble/manual
@(require "../common.rkt"
          (for-label (except-in racket/base
                                abs cos exp log sin sort sqrt max min length
                                + - * /)
                     torch
                     torch/nn))

@title[#:tag "layers"]{Layers}

A @deftech{layer} is a callable that owns tensors. The tensors it owns are
its @deftech{parameters}: the numbers training adjusts.

@section[#:tag "layers-calling"]{A layer is a procedure}

@racket[Linear] builds the familiar affine map. Apply it like any other
procedure:

@torch-examples[
(require torch torch/nn)
(define fc (Linear 3 2))
(layer? fc)
(shape (fc (zeros 1 3)))
]

@racket[named-parameters] lists what it owns, each with the name it will
carry in a checkpoint:

@torch-examples[
(map car (named-parameters fc))
]

@section[#:tag "layers-sequential"]{Composing them}

@racket[Sequential] chains layers and plain functions alike, so an
activation is a step like any other:

@torch-examples[
(define net (Sequential (Linear 4 8) relu (Linear 8 2)))
(shape (net (zeros 3 4)))
]

@section[#:tag "layers-define-layer"]{Declaring your own}

@racket[define-layer] is the analogue of subclassing @tt{nn.Module}. The
fields are the children, @racket[#:init] is the constructor body, and
@racket[#:forward] is what the layer computes. Child layers are callable by
name inside @racket[#:forward], the way @tt{self.fc1(x)} is in Python:

@torch-examples[
(define-layer mlp (fc1 fc2)
  #:init (d-in d-hidden d-out)
  (set! fc1 (Linear d-in d-hidden))
  (set! fc2 (Linear d-hidden d-out))
  #:forward (x)
  (~> x fc1 relu fc2))
(define model (mlp 4 16 1))
(shape (model (zeros 8 4)))
]

@racket[parameters] collects the tensors recursively, and
@racket[named-parameters] names them by the path down the tree:

@torch-examples[
(length (parameters model))
(map car (named-parameters model))
]

A model is an ordinary struct tree owned by the garbage collector. There is
no global parameter store to register with and nothing to free by hand:
drop the model and it goes away.

@section[#:tag "layers-modes"]{Training and evaluation modes}

Layers whose behaviour differs between fitting and predicting ---
@racket[Dropout] and the normalization layers --- read a mode flag.
@racket[train!] and @racket[eval!] set it, recursing through the tree, and
@racket[layer-training?] reads it back. Every layer starts in training
mode:

@torch-examples[
(layer-training? model)
(eval! model)
(layer-training? model)
(train! model)
(layer-training? model)
]
