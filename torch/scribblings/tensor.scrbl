#lang scribble/manual

@(require "common.rkt"
          (for-label racket/base
                     racket/contract
                     (only-in torch
                              arange device device? device/c dtype dtype/c
                              item manual-seed! ones randn shape sum tensor
                              tensor->list tensor-device tensor-dtype tensor?
                              to-dtype zeros)))

@title{Tensors}

@defmodule[torch #:link-target? #f]

A tensor is a rectangular array of numbers with one element type, resident
on one device.

@section{Construction}

@defproc[(tensor [data (or/c real? list? vector? f32vector? s64vector?
                             bytes?)]
                 [#:device device (or/c device/c #f) #f]
                 [#:dtype dtype (or/c 'float32 'int64 'uint8 #f) #f]
                 [#:requires-grad? requires-grad? boolean? #f])
         tensor?]{
Builds a tensor from Racket data: a number, or a list nested as deeply as
the tensor has dimensions. The nesting becomes the shape, and every row at
a given depth must be the same length. A @racket[bytes?] gives a
@racket['uint8] tensor, and an @racket[f32vector?] or @racket[s64vector?]
is the zero-copy path for bulk data already in the right layout.

Without @racket[#:dtype] the element type is inferred as PyTorch infers it
--- exact integers give @racket['int64], any inexact number gives
@racket['float32]:

@torch-examples[
(tensor '((1 2) (3 4)))
(tensor-dtype (tensor '(1.0 2.0)))
(tensor-dtype (tensor '(1.0 2.0) #:dtype 'int64))
]

Construction offers only the three dtypes above; reach the others, such as
@racket['float64], by casting an existing tensor with @racket[to-dtype].}

@defproc[(tensor? [v any/c]) boolean?]{
Returns @racket[#t] if @racket[v] is a tensor.}

@section{Creation by shape}

These take the dimensions as ordinary arguments rather than a list, and all
accept the same @racket[#:device], @racket[#:dtype] and
@racket[#:requires-grad?] keywords as @racket[tensor].

@racket[zeros] is documented with the placement keywords it shares with the
rest of the family, under @secref["Placement_at_construction"].

@torch-examples[(zeros 2 3)]

@defproc[(ones [dim exact-nonnegative-integer?] ...
               [#:device device device/c #f]
               [#:dtype dtype dtype/c #f]
               [#:requires-grad? requires-grad? boolean? #f])
        tensor?]{
As @racket[zeros], filled with ones. Unlike @racket[tensor], the creation
family accepts the whole of @racket[dtype/c].

@torch-examples[(ones 2 2)]}

@defproc[(randn [dim exact-nonnegative-integer?] ...
                [#:device device device/c #f]
                [#:dtype dtype (or/c 'float32 'float64) #f]
                [#:requires-grad? requires-grad? boolean? #f])
         tensor?]{
Draws each element independently from the standard normal distribution.
The dtype must be a floating-point one. Seed the generator with
@racket[manual-seed!] to repeat a draw.

@torch-examples[
(manual-seed! 0)
(randn 2 2)
]}

@defproc[(arange [start real?]
                 [end real? #f]
                 [step real? #f]
                 [#:device device device/c #f]
                 [#:dtype dtype dtype/c #f]
                 [#:requires-grad? requires-grad? boolean? #f])
         tensor?]{
A one-dimensional tensor counting up to but not including @racket[end], in
increments of @racket[step], like Python's @tt{range} and PyTorch's
@tt{torch.arange}. Given one argument it is the end and the count starts at
zero; given two, they are the start and the end.

It stays @racket['float32] by default, so
@racket[(arange n #:dtype 'int64)] is the index vector
@tt{torch.arange(n)} produces.

@torch-examples[
(arange 6)
(tensor->list (arange 0 6 2))
(tensor-dtype (arange 6 #:dtype 'int64))
]}

@defproc[(manual-seed! [seed exact-integer?]) void?]{
Seeds the global generator, so the draws that follow repeat. Seeded CPU
draws match PyTorch's for the same seed.}

@section{Queries}

@defproc[(shape [t tensor?]) (listof exact-nonnegative-integer?)]{
The dimensions, outermost first.

@torch-examples[(shape (tensor '((1 2 3) (4 5 6))))]}

@defproc[(dtype [t tensor?]) dtype/c]{
The element type: one of @racket['float32], @racket['float64],
@racket['int64], @racket['bool], @racket['uint8]. The short name, and the
one to prefer; it is Python's @tt{t.dtype}.

@torch-examples[
(dtype (tensor '(1 2)))
(dtype (tensor '(1.0 2.0)))
]}

@defproc[(tensor-dtype [t tensor?]) dtype/c]{
The same answer as @racket[dtype], under the name that says which kind of
value it reads.}

@defproc[(tensor-device [t tensor?]) device?]{
Where the storage lives: the same answer as @racket[device] given a tensor,
under the name that says which kind of value it reads. See
@secref["Devices_and_dtypes"].}

@section{Conversion out}

@defproc[(tensor->list [t tensor?]) list?]{
The elements as a flat Racket list, in row-major order, whatever the
tensor's shape.

@torch-examples[(tensor->list (tensor '((1 2) (3 4))))]}

@defproc[(item [t tensor?]) number?]{
The single element of a one-element tensor, as a Racket number. Raises if
the tensor holds anything other than exactly one element.

@torch-examples[(item (sum (tensor '((1 2) (3 4)))))]}
