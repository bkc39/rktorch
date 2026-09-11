#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     (only-in torch
                              cpu-device cuda-device device device? dtype
                              mps-device native-memory-use prop:to tensor
                              tensor-device tensor-dtype tensor? to to-able?
                              to-device to-dtype with-default-device ~>)
                     (only-in torch/nn
                              Buffer Linear Parameter adam buffers gen:layer
                              layer? load-state! parameters step!)))

@title{Devices and dtypes}

@defmodule[torch #:link-target? #f]

@defproc*[([(to [x (or/c tensor? to-able?)] [target (or/c device/c dtype/c)])
            (or/c tensor? to-able?)]
           [(to [x (or/c tensor? to-able?)] [target device/c] [dtype dtype/c])
            (or/c tensor? to-able?)])]{
PyTorch's @tt{.to}: moves @racket[x] to a device, casts it to a dtype, or
both in one native hop. A device target is any @racket[device/c] form,
@racket['cuda], @racket[(device 'cuda 1)], or a @racket[device?] value; a
dtype target is one of @racket['float32], @racket['float64],
@racket['int64], @racket['bool]. As in Python, the dtype may follow a device
target but a dtype target stands alone.

@racketblock[
(to x (device 'cuda))
(to x 'float64)
(to x 'cuda 'float32)
(~> x (to 'cuda))
]

For a tensor, the result is a new tensor and @racket[x] is untouched, except
that when nothing would change the result is @racket[x] itself, as
@tt{x.to("cpu") is x} in PyTorch. The identity matters for the
memory ledger: a second wrapper over aliased storage would be charged twice
(see @racket[native-memory-use]).

For a layer, or any value with @racket[prop:to], the move happens in place
and returns @racket[x]. Every layer from @racket[torch/nn] is
@racket[to-able?]: each of its @racket[parameters] and @racket[buffers] is
rebound to the moved storage under the same object, so references held by
the caller, by optimizers, and by the layer tree all stay valid; an
accumulated gradient moves along, and the parameter stays a
requires-grad leaf. Plain tensor fields are not moved, just as PyTorch
leaves plain tensor attributes where they are; register such a tensor with
@racket[Buffer] to have it follow the layer. A layer's dtype target must be
floating-point, @racket['float32] or @racket['float64], as
@tt{nn.Module.to} only accepts floating-point or complex dtypes.

Two rules carry over from PyTorch. Move the model before the first
optimizer @racket[step!]: @racket[adam] creates its moments on the
parameter's device at that step, and moments created earlier stay behind.
And a move that fails part-way, for example on a CUDA out-of-memory error,
leaves the layer with some parameters moved and some not.

@racketblock[
(define model (Linear 784 10))
(load-state! model "mnist.safetensors")
(to model (device 'cuda))
]

@racket[to-device] and @racket[to-dtype] are the single-axis primitives
underneath @racket[to]; both share its identity behaviour.
}

@defthing[prop:to struct-type-property?]{
A structure type property whose value is a procedure of three arguments,
the instance, a @racket[device?] or @racket[#f], and a dtype symbol or
@racket[#f], at least one of which is supplied. It is called by
@racket[to] to move the instance and must return the moved value. Layers
receive it through @racket[gen:layer]; a container of tensors outside the
layer system can implement it directly.
}

@defproc[(to-able? [v any/c]) boolean?]{
Recognises values carrying @racket[prop:to].
}

@defproc[(to-device [t tensor?] [dev device/c]) tensor?]{
Equivalent to @racket[(to t dev)].
}

@defproc[(to-dtype [t tensor?] [dtype dtype/c]) tensor?]{
Equivalent to @racket[(to t dtype)].
}

@section{Unsafe}

@defmodule[(submod torch/foreign unsafe)]

@defproc*[([(to! [t tensor?] [target (or/c device/c dtype/c)]) tensor?]
           [(to! [t tensor?] [target device/c] [dtype dtype/c]) tensor?])]{
The in-place primitive behind a layer move: rebinds @racket[t]'s storage to
the moved copy and returns @racket[t]. Every alias of @racket[t] observes the
change, which is why it lives in the unsafe submodule. The tensor's ledger
entry is re-accounted under the same handle, so @racket[native-memory-use]
reports the new device and byte count; the old storage is released by
libtorch at once when nothing else references it.
}
