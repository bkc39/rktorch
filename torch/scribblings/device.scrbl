#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     (only-in torch
                              arange cpu-device cuda-device device device/c
                              device? dtype dtype/c eye full full-like
                              mps-device native-memory-use ones ones-like
                              prop:to rand rand-like randn randn-like tensor
                              tensor-device tensor-dtype tensor? to to-able?
                              to-device to-dtype with-default-device zeros
                              zeros-like ~>)
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
@racket['float16], @racket['bfloat16], @racket['int64], @racket['bool],
@racket['uint8]. As in Python, the dtype may follow a device target but a
dtype target stands alone.

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
floating-point, one of @racket['float32], @racket['float64],
@racket['float16], @racket['bfloat16], as
@tt{nn.Module.to} only accepts floating-point or complex dtypes, and as
there it reaches only the floating-point parameters and buffers: an
@racket['int64] counter or a @racket['bool] mask registered with
@racket[Buffer] keeps its dtype and changes device alone.

Optimizer state follows: @racket[adam] creates its moments on the
parameter's device and dtype at the first @racket[step!], and a moment
created before a later move is brought to the parameter at the next step,
so a model may be moved at any point of training. One rule carries over
from PyTorch: a move that fails part-way, for example on a CUDA
out-of-memory error, leaves the layer with some parameters moved and some
not.

@racketblock[
(define model (Linear 784 10))
(load-state! model "mnist.safetensors")
(to model (device 'cuda))
]

@racket[to-device] and @racket[to-dtype] are the single-axis primitives
underneath @racket[to]; both share its identity behaviour.
}

@defthing[device/c contract?]{
A device designator: a @racket[device?] value, one of @racket['cpu],
@racket['cuda], @racket['mps], or @racket[(list 'cuda index)].
}

@defthing[dtype/c contract?]{
One of @racket['float32], @racket['float64], @racket['float16],
@racket['bfloat16], @racket['int64], @racket['bool], @racket['uint8].
}

@section{Half precision}

The 16-bit floats are dtypes like any other: @racket['float16] is IEEE
half, with ten mantissa bits and a largest value of 65504, and
@racket['bfloat16] keeps float32's exponent with seven mantissa bits, so it
holds float32's range at a quarter of the precision. Every constructor
takes them, @racket[to] casts to and from them, a layer moves to them, and
values read back through float32, so @racket[tensor->list] and
@racket[item] are exact for what the tensor holds. The safetensors
container writes them as @tt{F16} and @tt{BF16}.

Training in half precision is done the way PyTorch does it: the parameters
stay @racket['float32] and the forward runs under autocast, which casts the
matrix multiplications and convolutions to the half dtype and keeps the
reductions, losses and normalisations in float32, per PyTorch's cast lists.
The 3090 Ti and its generation run @racket['bfloat16] on tensor cores with
float32's range, so no loss scaling is needed; @racket['float16] is the
choice for inference and storage.

@racketblock[
(for ([(xb yb) (in-dataloader loader)])
  (zero-grads! opt)
  (define loss
    (with-autocast #:device 'cuda
      (cross-entropy (net xb) yb)))
  (backward! loss)
  (step! opt))
]

@defform[(with-autocast maybe-device maybe-dtype body ...+)
         #:grammar [(maybe-device (code:line) (code:line #:device device))
                    (maybe-dtype (code:line) (code:line #:dtype dtype))]]{
Runs the body with autocast on for @racket[device], which is a device type
or a @racket[device?] value and defaults to the default device, in
@racket[dtype], @racket['bfloat16] unless given @racket['float16]. The state
is per thread and per device type, as in @tt{torch.autocast}, and leaving
the body puts back whatever was there before, so the form nests. Run
@racket[backward!] outside the form, as PyTorch recommends: the gradients
arrive in the parameters' own dtype either way.
}

@defproc[(call-with-autocast [thunk (-> any)]
                             [#:device device (or/c 'cpu 'cuda 'mps device?)
                              (default-device)]
                             [#:dtype dtype (or/c 'float16 'bfloat16) 'bfloat16])
         any]{
The procedure form of @racket[with-autocast].
}

@defproc[(autocast-enabled? [device (or/c 'cpu 'cuda 'mps device?) (default-device)])
         boolean?]{
Whether autocast is on for @racket[device] on the calling thread.
}

@defproc[(autocast-dtype [device (or/c 'cpu 'cuda 'mps device?) (default-device)])
         (or/c 'float16 'bfloat16)]{
The dtype autocast casts to on @racket[device], set or not; the process
default is @racket['float16] for CUDA and @racket['bfloat16] for the CPU.
}

@defproc[(tensor->bytes [t tensor?]) bytes?]{
The element bytes of @racket[t] as they are, row-major in its own dtype and
the host's byte order: the safetensors payload, and the only way a 16-bit
float leaves the process without widening.
}

@defproc[(bytes->tensor [bs bytes?] [dtype dtype/c]
                        [shape (listof exact-nonnegative-integer?)])
         tensor?]{
The inverse of @racket[tensor->bytes]: a tensor of @racket[dtype] and
@racket[shape] over a copy of @racket[bs], whose length must be the
element count times the element size. It lands on the default device.
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

@section{Placement at construction}

@defproc*[([(zeros [dim exact-nonnegative-integer?] ...
                   [#:device device device/c] [#:dtype dtype dtype/c]
                   [#:requires-grad? requires-grad? boolean? #f]) tensor?]
           [(zeros [dims (listof exact-nonnegative-integer?)]
                   [#:device device device/c] [#:dtype dtype dtype/c]
                   [#:requires-grad? requires-grad? boolean? #f]) tensor?])]{
A tensor of zeros, with the dims as rest arguments or as one list, as
@tt{torch.zeros(2, 3)} and @tt{torch.zeros((2, 3))}. The device and dtype
default to the process default device and @racket['float32]; when given they
are chosen at native construction, so a tensor never takes a hop through
another device on its way to where it will live. With
@racket[#:requires-grad?] the result is marked as a leaf after construction,
which an integer dtype refuses as PyTorch does. @racket[ones], @racket[full],
@racket[randn], and @racket[rand] take the same arguments; the two random
constructors accept only a floating-point dtype and draw from the chosen
device's generator.

@racket[arange] and @racket[eye] take the same three keywords after their
positional arguments. @racket[arange] stays @racket['float32] by default, as
before; @racket[(arange n #:dtype 'int64)] is the index vector
@tt{torch.arange(n)} produces.
}

@defproc[(zeros-like [t tensor?]
                     [#:device device device/c] [#:dtype dtype dtype/c]
                     [#:requires-grad? requires-grad? boolean? #f])
         tensor?]{
Zeros with @racket[t]'s shape, device, and dtype unless overridden, as
@tt{torch.zeros_like}. @racket[ones-like], @racket[full-like] (which takes
the fill value after @racket[t]), @racket[randn-like], and @racket[rand-like]
are the same for their constructors. Optimizer state is the typical use: a
moment created by @racket[zeros-like] lives where its parameter does,
however the parameter got there.
}

@section{Unsafe}

@defmodule[(submod torch/foreign unsafe)]

@defproc*[([(to! [t tensor?] [target (or/c device/c dtype/c)]) tensor?]
           [(to! [t tensor?] [target device/c] [dtype dtype/c]) tensor?])]{
The in-place primitive behind a layer move: rebinds @racket[t]'s storage to
the moved copy and returns @racket[t]. Every reference to @racket[t]
observes the change, which is why it lives in the unsafe submodule; a view
or detached copy made earlier is a separate tensor and keeps the old
storage, device, and dtype. The tensor's ledger
entry is re-accounted under the same handle, so @racket[native-memory-use]
reports the new device and byte count; the old storage is released by
libtorch at once when nothing else references it.
}
