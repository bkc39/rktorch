#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     (only-in torch cuda-if-available device? randn-like tensor?)
                     torch/data/loader
                     (only-in torch/nn Conv2d ConvTranspose2d GroupNorm Linear
                              define-layer)
                     torch/vision/cifar10
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
                       [#:device device (or/c #f device?) #f])
         (values tensor? tensor?)]{
The split's images as a float32 tensor of shape @tt{[N 3 32 32]} with
pixels in @tt{[-1, 1]}, and its labels as an int64 tensor of shape
@tt{[N]}, the layout a diffusion model trains on, built on
@racket[device] or else the default device. The five training batches
come back as one tensor.
}

@defproc[(cifar10-dataset [split (or/c 'train 'test) 'train]
                          [#:device device (or/c #f device?) #f])
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
                                   [#:device device (or/c #f device?) #f])
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
                          [#:offset offset (>=/c 0) 0.008])
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

@defproc[(q-sample [s schedule?] [x0 tensor?] [t tensor?] [noise tensor?])
         tensor?]{
@tt{q(x_t | x_0)} in closed form: with @tt{a} the cumulative product at each
image's timestep @racket[t], an int64 tensor of shape @tt{[N]},
@tt{sqrt(a) x0 + sqrt(1 - a) noise}. @racket[noise] is drawn by the caller,
typically @racket[randn-like], so a seeded run replays.
}

@defproc[(sinusoidal-embedding [t tensor?] [dim even-positive-integer?])
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
                   [t-dim exact-positive-integer?])
         res-block?]{
The UNet's block: @racket[GroupNorm] of eight groups, silu, a 3x3
@racket[Conv2d], the time embedding projected and added per channel, a
second norm, silu and convolution, plus the residual through a 1x1
convolution when the widths differ. Widths are multiples of eight.
}

@defproc[(UNet [#:base base channels/c 32]) unet?]{
A two-level UNet for 32x32 RGB images: a @racket[TimeEmbedding] of
@racket[base] features, an input convolution to @racket[base] channels, a
@racket[ResBlock] and a stride-2 convolution per level down to
@tt{2base} channels at 8x8, a middle block, then a @racket[ConvTranspose2d]
per level back up with the matching skip concatenated, and an output
convolution to three channels. Called as @racket[(net x t)] it returns the
noise estimate for @racket[x] at timesteps @racket[t].
}
