#lang scribble/manual

@(require "common.rkt"
          (for-label racket/base
                     racket/contract
                     (only-in torch dtype flatten log-softmax max-pool2d relu
                              shape tensor tensor? zeros)
                     (only-in torch/nn
                              Conv1d Conv2d ConvTranspose2d Dropout Embedding
                              Flatten GroupNorm LayerNorm Linear MaxPool2d
                              Parameter Sequential batch-norm1d? batch-norm2d?
                              binary-cross-entropy-with-logits conv2d?
                              cross-entropy ctc-loss define-layer dropout?
                              kaiming-uniform layer? linear? load-state!
                              max-pool2d? mse-loss named-buffers
                              named-parameters parameters save-state!
                              uniform-init)))

@title{Built-in layers and losses}

@defmodule[torch/nn #:link-target? #f]

The concrete layers and losses the library ships. @racket[define-layer],
the interface the layers implement, and the container forms are described
in @secref["Layers"]; the optimizers and learning-rate schedules that
train them are in @secref["optimizers"].

@section{Layer constructors}

Every layer is applied like a procedure, and is @racket[layer?].
@racket[Sequential], which composes them, is described with the other
container forms in @secref["Layers"].

@defproc[(Linear [in exact-positive-integer?] [out exact-positive-integer?])
         layer?]{
The affine map @tt{xW@superscript{T} + b}, from @racket[in] features to
@racket[out]. Owns @racket["weight"] and @racket["bias"], initialized as
PyTorch's @tt{nn.Linear} initializes them.

@torch-examples[
(define fc (Linear 3 2))
(shape (fc (zeros 1 3)))
(map car (named-parameters fc))
]}

@defproc[(Conv2d [in exact-positive-integer?]
                 [out exact-positive-integer?]
                 [kernel (or/c exact-positive-integer?
                               (list/c exact-positive-integer?
                                       exact-positive-integer?))]
                 [#:stride stride (or/c exact-positive-integer?
                                        (list/c exact-positive-integer?
                                                exact-positive-integer?))
                           1]
                 [#:padding padding (or/c exact-nonnegative-integer?
                                          (list/c exact-nonnegative-integer?
                                                  exact-nonnegative-integer?))
                            0])
         layer?]{
Two-dimensional convolution over a batch shaped @tt{[N, in, H, W]},
answering @tt{[N, out, H', W']} with the usual convolution arithmetic.

Each size is an integer for a square one, or a two-element list
@racket[(list height width)] for an asymmetric one, mirroring PyTorch's
@tt{nn.Conv2d(kernel_size=(3, 5))}.

@torch-examples[
(shape ((Conv2d 1 4 3) (zeros 1 1 8 8)))
]}

@defproc[(Dropout [#:p p (and/c real? (>=/c 0) (</c 1)) 0.5]) layer?]{
Zeroes each element independently with probability @racket[p] while the
layer is in training mode, and is the identity while it is evaluating. See
@secref["Layers"] for the mode flag.}

@section{Collecting parameters}

@defproc[(parameters [m layer?]) (listof tensor?)]{
Every parameter in the layer tree, depth first, each appearing once.}

@defproc[(named-parameters [m layer?] [prefix string? ""])
         (listof (cons/c string? tensor?))]{
As @racket[parameters], each paired with its dotted path through the tree
--- @racket["fc1.weight"] --- which is the name it carries in a
checkpoint.

@racket[prefix] is prepended verbatim, so a caller supplying one includes
its own separator: @racket["enc."] gives @racket["enc.fc1.weight"].

@torch-examples[
(map car (named-parameters (Sequential (Linear 2 2))))
]}

@defproc[(layer? [v any/c]) boolean?]{
Whether @racket[v] implements the layer interface.}

@section{Losses}

@defproc[(mse-loss [input tensor?] [target tensor?]) tensor?]{
Mean squared error between two tensors of the same shape, as a one-element
tensor.}

@defproc[(cross-entropy [logits tensor?] [targets tensor?]) tensor?]{
The classification loss: @racket[log-softmax] of @tt{[N C]} logits over
the classes, picked out at the @tt{[N]} integer @racket[targets], averaged
over the batch. Takes raw logits; do not apply a softmax first.}

@defproc[(binary-cross-entropy-with-logits [logits tensor?] [targets tensor?]
                                           [#:weight weight (or/c tensor? #f) #f]
                                           [#:pos-weight pos-weight (or/c tensor? #f) #f])
         tensor?]{
The binary loss on raw logits of any shape against @racket[targets] of the
same shape holding zeros and ones --- the sigmoid folded in for numerical
stability, which is why a discriminator's head has no sigmoid of its own.
@racket[weight] scales each element and @racket[pos-weight] each positive.}

@defproc[(ctc-loss [log-probs tensor?] [targets tensor?]
                   [#:input-lengths input-lengths (non-empty-listof exact-positive-integer?)]
                   [#:target-lengths target-lengths (non-empty-listof exact-nonnegative-integer?)]
                   [#:blank blank exact-nonnegative-integer? 0]
                   [#:zero-infinity? zero-infinity? boolean? #f])
         tensor?]{
Connectionist temporal classification: @racket[log-probs] is @tt{[T N C]}
log-softmaxed frames, @racket[targets] the @tt{[N S]} label sequences, and
the two length lists say how much of each is real. Averaged over the batch
as PyTorch's default is; @racket[#:zero-infinity?] zeroes a loss that no
alignment can reach rather than propagating an infinity.}

@section{More layers}

@defproc[(Conv1d [in exact-positive-integer?]
                 [out exact-positive-integer?]
                 [kernel (or/c exact-positive-integer?
                               (list/c exact-positive-integer?))]
                 [#:stride stride (or/c exact-positive-integer?
                                        (list/c exact-positive-integer?)) 1]
                 [#:padding padding (or/c exact-nonnegative-integer?
                                          (list/c exact-nonnegative-integer?)) 0]
                 [#:dilation dilation (or/c exact-positive-integer?
                                            (list/c exact-positive-integer?)) 1])
         layer?]{
One-dimensional convolution over an @tt{[N in L]} batch, answering
@tt{[N out L']}; the front end of a speech model runs this over
spectrogram frames.}

@defproc[(ConvTranspose2d [in exact-positive-integer?]
                          [out exact-positive-integer?]
                          [kernel (or/c exact-positive-integer?
                                        (list/c exact-positive-integer?
                                                exact-positive-integer?))]
                          [#:stride stride (or/c exact-positive-integer?
                                                 (list/c exact-positive-integer?
                                                         exact-positive-integer?))
                                    1]
                          [#:padding padding (or/c exact-nonnegative-integer?
                                                   (list/c exact-nonnegative-integer?
                                                           exact-nonnegative-integer?))
                                     0]
                          [#:output-padding output-padding
                           (or/c exact-nonnegative-integer?
                                 (list/c exact-nonnegative-integer?
                                         exact-nonnegative-integer?))
                           0]
                          [#:dilation dilation (or/c exact-positive-integer?
                                                     (list/c exact-positive-integer?
                                                             exact-positive-integer?))
                                      1]
                          [#:groups groups exact-positive-integer? 1])
         layer?]{
The transposed convolution that upsamples, @tt{nn.ConvTranspose2d}: a
DCGAN generator is a stack of these from a latent vector up to an image.
@racket[output-padding] resolves the ambiguity in the output size that a
stride above one leaves.}

@defproc[(MaxPool2d [kernel (or/c exact-positive-integer?
                                  (list/c exact-positive-integer?
                                          exact-positive-integer?))]
                    [#:stride stride (or/c #f exact-positive-integer?
                                           (list/c exact-positive-integer?
                                                   exact-positive-integer?))
                              #f]
                    [#:padding padding (or/c exact-nonnegative-integer?
                                             (list/c exact-nonnegative-integer?
                                                     exact-nonnegative-integer?))
                               0])
         layer?]{
The layer form of @racket[max-pool2d]; the stride defaults to the kernel
size. It owns no parameters.}

@defproc[(Flatten [#:start-dim start-dim exact-integer? 1]
                  [#:end-dim end-dim exact-integer? -1])
         layer?]{
The layer form of @racket[flatten], merging the axes from
@racket[start-dim] on, so a batch of feature maps becomes a batch of
vectors on the way into a @racket[Linear] head. It owns no parameters.}

@defproc[(Embedding [count exact-positive-integer?] [dim exact-positive-integer?])
         layer?]{
A lookup table of @racket[count] rows of @racket[dim] features, normally
initialized; applied to an int64 tensor of indices it answers the rows,
with @racket[dim] appended to the input's shape. Tokens go in, vectors
come out.

@torch-examples[
(shape ((Embedding 10 4) (tensor '((1 2 3)))))
]}

@defproc[(LayerNorm [normalized-shape (or/c exact-positive-integer?
                                            (non-empty-listof exact-positive-integer?))]
                    [#:eps eps real? 1e-5])
         layer?]{
Normalizes each input over its trailing @racket[normalized-shape]
dimensions to zero mean and unit variance, then scales and shifts by a
learned @racket["weight"] and @racket["bias"] of that shape; the
normalization inside a transformer block.}

@defproc[(GroupNorm [groups exact-positive-integer?]
                    [channels exact-positive-integer?]
                    [#:eps eps real? 1e-5])
         layer?]{
Normalizes an @tt{[N C H W]} batch within each of @racket[groups] groups
of channels, @racket[channels] divisible by @racket[groups]; the
normalization the diffusion UNet uses, independent of the batch size.}

@deftogether[(@defproc[(linear? [v any/c]) boolean?]
              @defproc[(conv2d? [v any/c]) boolean?]
              @defproc[(max-pool2d? [v any/c]) boolean?]
              @defproc[(dropout? [v any/c]) boolean?]
              @defproc[(batch-norm1d? [v any/c]) boolean?]
              @defproc[(batch-norm2d? [v any/c]) boolean?])]{
The predicates, lowercase per Racket idiom where the constructors are
PascalCase per PyTorch's.}

@section{Initializers}

The functions the constructors use to draw their initial parameters,
exposed for a layer written with @racket[define-layer]. Each answers a
fresh tensor on the default device, consuming the global random stream
exactly as PyTorch's counterpart does, so seeded initialization matches.

@defproc[(uniform-init [dims (listof exact-nonnegative-integer?)]
                       [low real?] [high real?])
         tensor?]{
Uniform on @tt{[low, high)}.}

@defproc[(kaiming-uniform [dims (listof exact-nonnegative-integer?)]
                          [#:a a real? (sqrt 5.0)])
         tensor?]{
Kaiming's uniform initialization with the bound PyTorch's
@tt{nn.Linear} and @tt{nn.Conv2d} use by default, the fan-in read from
@racket[dims].}

@section{Checkpoints}

@defproc[(save-state! [model layer?] [path path-string?]) void?]{
Writes @racket[model]'s parameters and buffers to @racket[path] as a
safetensors file, each under its dotted name from
@racket[named-parameters] and @racket[named-buffers].}

@defproc[(load-state! [model layer?] [path path-string?]) void?]{
Loads the safetensors file at @racket[path] into @racket[model]'s
parameters and buffers, matched by their dotted names, in place. The
file must name every tensor the model has and no others.}
