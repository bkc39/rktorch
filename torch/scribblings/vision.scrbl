#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     (only-in torch cuda-if-available device/c draw-seed generator?
                              make-generator randn-like tensor?
                              upsample-nearest2d)
                     torch/data/loader
                     (only-in torch/nn Conv2d Dropout Embedding GroupNorm Linear
                              define-layer)
                     torch/vision/cifar10
                     torch/vision/transforms
                     torch/vision/diffusion))

@title{Vision datasets}

@defmodule[torch/vision/cifar10]

CIFAR-10 from its binary distribution: 60000 colour images of 32 by 32
pixels in ten classes, 50000 for training and 10000 for testing. The
archive is fetched once into the cache directory (or
@envvar{RKTORCH_CIFAR10_DIR}), decoded in full before it is kept so a
redirect page or a transfer cut short never reaches the cache, and unpacked
in memory; nothing else is written.

@racketblock[
(define loader
  (dataloader (cifar10-dataset 'train #:device (cuda-if-available))
              #:batch-size 128 #:shuffle? #t))
]

@defproc[(load-cifar10 [split (or/c 'train 'test) 'train]
                       [#:device device (or/c #f device/c) #f])
         (values tensor? tensor?)]{
The split's images as a float32 tensor of shape @tt{[N 3 32 32]} with
pixels in @tt{[-1, 1]}, and its labels as an int64 tensor of shape
@tt{[N]}, the layout a diffusion model trains on, built on
@racket[device] or else the default device. The five training batches
come back as one tensor.
}

@defproc[(cifar10-dataset [split (or/c 'train 'test) 'train]
                          [#:device device (or/c #f device/c) #f])
         dataset?]{
@racket[load-cifar10] as a @racket[tensor-dataset], so a loader over it
batches where the tensors live.
}

@defthing[cifar10-label-names (listof string?)]{
The ten class names in label order, @racket["airplane"] through
@racket["truck"], as the archive's @tt{batches.meta.txt} lists them.
}

@defproc[(load-cifar10-fixture) (values tensor? tensor?)]{
The first 256 records of the first training batch, committed with the
package, in the same form as @racket[load-cifar10]; for tests and offline
examples.
}

@defproc[(cifar10-records->tensors [bs bytes?]
                                   [#:device device (or/c #f device/c) #f])
         (values tensor? tensor?)]{
Parses a buffer of 3073-byte records, one label byte followed by the
red, green and blue planes of an image, into the tensors above, on
@racket[device] or else the default device. A buffer that is not a whole
number of records is an error.
}

@defproc[(cifar10-archive-files) (listof (cons/c string? bytes?))]{
Every file of the cached archive by base name, unpacked in memory,
fetching the archive first when the cache lacks it.
}

@defproc[(cifar10-cached?) boolean?]{
Whether the archive is already in the cache, so a caller can decide
whether to trigger the fetch; the tests only touch a cached archive.
}

@defproc[(tar-entries [bs bytes?]) (listof (cons/c string? bytes?))]{
The regular files of an uncompressed tar buffer, name and contents,
enough for the archives the datasets ship in.
}

@section{Diffusion}

@defmodule[torch/vision/diffusion]

The pieces of a denoising diffusion model in the epsilon-prediction form:
a schedule of noise levels, the closed-form jump to any timestep, and a
small UNet. The training loop and the sampler are the examples' business.

@defproc[(linear-schedule [steps exact-positive-integer? 1000]
                          [#:beta-start beta-start (real-in 0 1) 1e-4]
                          [#:beta-end beta-end (real-in 0 1) 0.02])
         schedule?]{
The DDPM schedule: @racket[steps] variances evenly spaced from
@racket[beta-start] to @racket[beta-end], with their alphas and cumulative
products as float32 tensors on the default device, so build it where the
model lives.
}

@defproc[(cosine-schedule [steps exact-positive-integer? 1000]
                          [#:offset offset finite-nonnegative-real? 0.008])
         schedule?]{
The improved-DDPM schedule: cumulative products following a squared cosine
of the timestep, each variance capped at @racket[0.999].
}

@deftogether[(@defproc[(schedule? [v any/c]) boolean?]
              @defproc[(schedule-steps [s schedule?]) exact-positive-integer?]
              @defproc[(schedule-betas [s schedule?]) tensor?]
              @defproc[(schedule-alphas [s schedule?]) tensor?]
              @defproc[(schedule-alpha-bars [s schedule?]) tensor?])]{
A schedule and its tables, each of shape @tt{[steps]}.
}

@defproc[(q-sample [s schedule?] [x0 tensor?] [t int64-vector?] [noise tensor?])
         tensor?]{
@tt{q(x_t | x_0)} in closed form: with @tt{a} the cumulative product at each
image's timestep @racket[t], an int64 tensor of shape @tt{[N]},
@tt{sqrt(a) x0 + sqrt(1 - a) noise}. @racket[noise] is drawn by the caller,
typically @racket[randn-like], so a seeded run replays.
}

@defproc[(sinusoidal-embedding [t int64-vector?] [dim even-positive-integer?])
         tensor?]{
Timesteps @racket[t], shape @tt{[N]}, as @tt{[N dim]} sinusoidal features:
the sine half then the cosine half over frequencies falling geometrically
from @tt{1} to @tt{1/10000}.
}

@defproc[(TimeEmbedding [dim even-positive-integer?]) time-embedding?]{
A layer mapping timesteps to a @tt{[N 4dim]} embedding: the sinusoidal
features through two @racket[Linear] layers with a silu between.
}

@defproc[(ResBlock [in channels/c] [out channels/c]
                   [t-dim exact-positive-integer?]
                   [#:dropout dropout (real-in 0 1) 0])
         res-block?]{
The UNet's block: @racket[GroupNorm] of 32 groups, silu, a 3x3
@racket[Conv2d], the time embedding projected and added per channel, a
second norm, silu, @racket[Dropout] and convolution, plus the residual
through a 1x1 convolution when the widths differ. Widths are multiples of
32, the group count.
}

@defproc[(AttentionBlock [channels channels/c]) attention-block?]{
Single-head self-attention over a feature map, the DDPM form: after a
norm each pixel's channels are one token, @racket[Linear] maps give the
queries, keys and values, softmax over the scaled dot products mixes the
tokens, a @racket[Linear] projection follows, and the result is added to
the input. The convolutional path before the block is the encoder that
turns pixels into these tokens.
}

@defproc[(Downsample [channels channels/c]) downsample?]{
A stride-2 3x3 convolution; called as @racket[(down x temb)] so it slots
into the down path beside the blocks, the embedding ignored.
}

@defproc[(Upsample [channels channels/c]) upsample?]{
Nearest-neighbour doubling through @racket[upsample-nearest2d] followed
by a 3x3 convolution, the DDPM upsampling that avoids the checkerboard of
a transposed convolution.
}

@defproc[(UNet [#:base base channels/c 128]
               [#:mults mults (listof exact-positive-integer?) '(1 2 2 2)]
               [#:blocks blocks exact-positive-integer? 2]
               [#:attention attention (listof exact-positive-integer?) '(16)]
               [#:dropout dropout (real-in 0 1) 0.1]
               [#:classes classes (or/c #f exact-positive-integer?) #f])
         unet?]{
The DDPM UNet for 32x32 RGB images, the paper's CIFAR-10 configuration
by default: one resolution level per entry of @racket[mults], at most
five of them from 32x32 down to 2x2, each @racket[base] times that entry
wide and half the resolution of the last,
@racket[blocks] @racket[ResBlock]s per level on the way down and one more
per level on the way up, each followed by an @racket[AttentionBlock] at
the resolutions listed in @racket[attention], each of which must be one
of the levels' resolutions, a @racket[Downsample]
between levels going down and an @racket[Upsample] coming up, a middle
of block, attention, block, and a @racket[TimeEmbedding] of @racket[base]
features. Every block's output on the way down is concatenated back in on
the way up. With @racket[classes] the network is class-conditional: an
@racket[Embedding] of that many labels plus one null label, added to the
time embedding, so a label of @racket[classes] means "no label" for
classifier-free guidance. Called as @racket[(net x t y)] it returns the
noise estimate for @racket[x] at timesteps @racket[t] and int64 labels
@racket[y], which is @racket[#f] for an unconditional network. The
default network has 35.7 million parameters.
}

@defproc[(unet-classes [net unet?]) (or/c #f exact-positive-integer?)]{
The class count @racket[net] was built with, also its null label, or
@racket[#f] for an unconditional network.
}

@section{Transforms}

@defmodule[torch/vision/transforms]

Augmentation on an image batch where it lives: each transform takes an
@tt{[N C H W]} tensor and returns one of the same shape on the same
device. The random choices are per image, drawn on the host from a Racket
generator that one @racket[draw-seed] from the torch generator seeds per
batch, so a loader built on @racket[(make-generator 0)] replays its
augmentation as well as its batch order. They are the transform's own
draws: a torchvision pipeline on the same seed picks different crops.

@racketblock[
(define g (make-generator 0))
(define loader
  (dataloader (cifar10-dataset 'train #:device (cuda-if-available))
              #:batch-size 128 #:shuffle? #t #:generator g))
(for ([(xb yb) (in-dataloader loader)])
  (define augmented
    (random-horizontal-flip (random-crop xb #:padding 4 #:generator g)
                            #:generator g))
  ...)
]

@defproc[(random-horizontal-flip [x tensor?]
                                 [#:p p (real-in 0 1) 0.5]
                                 [#:generator generator (or/c generator? #f) #f])
         tensor?]{
Mirrors each image of the rank-4 batch @racket[x] along its width with
probability @racket[p]; the rest pass through unchanged.
}

@defproc[(random-crop [x tensor?]
                      [#:padding padding exact-nonnegative-integer? 4]
                      [#:generator generator (or/c generator? #f) #f])
         tensor?]{
Pads every image of the rank-4 batch @racket[x] with @racket[padding] zero
pixels on each side and cuts a window of the original size at a per-image
offset, torchvision's @tt{RandomCrop(32, padding=4)} for CIFAR-10. With
@racket[padding] 0 it is the identity.
}
