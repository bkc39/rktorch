#lang scribble/manual

@(require "common.rkt"
          (for-label racket/base
                     racket/contract
                     (only-in torch cuda-if-available device/c draw-seed generator?
                              make-generator matmul randn-like tensor?
                              to-dtype upsample-nearest2d)
                     torch/data/loader
                     (only-in torch/nn Conv2d Dropout Embedding GroupNorm Linear
                              define-layer load-state!)
                     torch/vision/cifar10
                     torch/vision/image
                     torch/vision/ppm
                     torch/vision/resnet
                     torch/vision/transforms
                     torch/vision/weights
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

The preprocessing a pretrained network expects, and augmentation on an
image batch where it lives. Every transform returns a tensor on its
input's device.

@torch-examples[
(require torch/vision/image torch/vision/transforms)
(define photo
  (read-image (collection-file-path "smooth-401x299.jpg"
                                    "torch" "vision" "fixtures" "images")))
(shape photo)
(define x
  (imagenet-normalize
   (center-crop (resize (convert-image-dtype photo) 256) 224)))
(shape x)
]

The first four are torchvision's @tt{transforms.functional} of the same
names and take an image, @tt{[C H W]}, or a batch, @tt{[N C H W]}.

@defproc[(convert-image-dtype [x tensor?]
                              [dtype (or/c 'uint8 'float16 'bfloat16
                                           'float32 'float64)
                                     'float32])
         tensor?]{
@racket[x] as @racket[dtype], rescaled between the two conventions for
pixels: a @racket['uint8] image holds 0 to 255, a float one 0 to 1. So a
decoded image divides by 255 on the way to a float dtype, and a float
image multiplies by @racket[255.999] and truncates on the way back, which
lands 1.0 on 255 and nothing past it. Between two float dtypes it is
@racket[to-dtype]; to its own dtype it is @racket[x] itself.
}

@defproc[(resize [x tensor?]
                 [size (or/c exact-positive-integer?
                             (list/c exact-positive-integer?
                                     exact-positive-integer?))]
                 [#:antialias? antialias? boolean? #t])
         tensor?]{
Resamples the float image or batch @racket[x] to @racket[size]
bilinearly. A pair is the new height and width; a single number is the
new length of the shorter side, the longer one scaled to keep the aspect
ratio and truncated, so @tt{[3 299 401]} resized to 256 is
@tt{[3 256 343]}. With @racket[antialias?], the default as in
torchvision, a downscale widens the triangle filter to the scale factor
the way Pillow does, so every input pixel contributes; without it each
output pixel interpolates the two inputs nearest its centre, which aliases
when shrinking by more than two. Either way the result is
@tt{F.interpolate(mode="bilinear", align_corners=False)} to within float
rounding.

The resampling is two matrix products, one per axis, with weights
computed on the host, rather than ATen's upsampling kernel: it is
differentiable, it runs on any device @racket[matmul] does, and it needs
no optional-float argument, which the generated surface does not yet
marshal. The half dtypes are resampled in float32 and narrowed back. A
@racket['uint8] image is a contract violation: convert it first, as
above.
}

@defproc[(center-crop [x tensor?]
                      [size (or/c exact-positive-integer?
                                  (list/c exact-positive-integer?
                                          exact-positive-integer?))])
         tensor?]{
The central @racket[size] window of @racket[x], of any dtype, a view on
its storage. A pair is the height and width, a single number a square.
The window's offset is half the difference rounded to even, as Python's
@tt{round} does in torchvision. A window larger than the image is a
contract violation, where torchvision would pad.
}

@defproc[(normalize [x tensor?]
                    [mean (listof real?)]
                    [std (listof (and/c real? positive?))])
         tensor?]{
Subtracts @racket[mean] and divides by @racket[std], one value per
channel of the float image or batch @racket[x].
}

@deftogether[(@defthing[imagenet-mean (listof real?)]
              @defthing[imagenet-std (listof real?)])]{
The per-channel statistics the torchvision ImageNet weights were trained
with, @racket['(0.485 0.456 0.406)] and @racket['(0.229 0.224 0.225)].
}

@defproc[(imagenet-normalize [x tensor?]) tensor?]{
@racket[(normalize x imagenet-mean imagenet-std)] for a three-channel
float image or batch in @tt{[0, 1]}, the input every pretrained
torchvision classifier expects after @racket[resize] to 256 and
@racket[center-crop] to 224.
}

The two random transforms take an @tt{[N C H W]} batch and return one of
the same shape. The random choices are per image, drawn on the host from a Racket
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

@section{ResNet}

@defmodule[torch/vision/resnet]

The residual network for 32x32 images: a 3x3 stem in place of ImageNet's
7x7 stem and max-pool, four stages of basic blocks doubling the width and
halving the resolution, global average pooling and a linear head. Every
convolution is bias-free, as its batch norm absorbs the bias. The training
loop is @filepath{examples/racket/09-resnet.rkt}.

@defproc[(BasicBlock [in exact-positive-integer?]
                     [out exact-positive-integer?]
                     [#:stride stride exact-positive-integer? 1])
         basic-block?]{
Two 3x3 convolutions with batch norm and a ReLU between, added to the
input and passed through a ReLU; when the stride or the width changes, a
1x1 convolution with batch norm projects the input first, and otherwise the
shortcut is the identity with no parameters of its own, as torchvision.
}

@defproc[(ResNet [#:classes classes exact-positive-integer? 10]
                 [#:base base exact-positive-integer? 64]
                 [#:blocks blocks (list/c exact-positive-integer?
                                          exact-positive-integer?
                                          exact-positive-integer?
                                          exact-positive-integer?)
                                   '(2 2 2 2)])
         resnet?]{
The stem at @racket[base] channels, then four stages of @racket[blocks]
basic blocks at @racket[base], twice, four and eight times that, the last
three at stride 2, and the head. The default is ResNet-18 as the CIFAR-10
literature shapes it, 11.2 million parameters; @racket[(ResNet #:base 16)]
is the narrow one the tests and the parity twin train. Called on an
@tt{[N 3 32 32]} batch it returns @tt{[N classes]} logits; any other rank
or channel count is a contract violation, blamed on the caller.
}

@deftogether[(@defproc[(basic-block? [v any/c]) boolean?]
              @defproc[(resnet? [v any/c]) boolean?])]{
The predicates.
}

@section{Reading images}

@defmodule[torch/vision/image]

JPEG and PNG decoding into a @racket['uint8] tensor of shape
@tt{[C H W]}, torchvision's @tt{decode_image} layout. The decoder is
@hyperlink["https://github.com/nothings/stb"]{stb_image}, a header-only
library compiled into the native library with only those two formats, so
reading an image needs no library at run time.

A PNG decodes to exactly the pixels torchvision's decoder returns. A
JPEG differs from libjpeg-turbo's decoding by a count or two in places,
because the two decoders use different inverse transforms and chroma
upsampling. Baseline and progressive JPEGs decode; arithmetic-coded and
12-bit ones do not. A 16-bit PNG is reduced to 8 bits. EXIF orientation
is ignored, as it is by @tt{decode_image} by default.

@defproc[(decode-image [bs bytes?]
                       [#:mode mode (or/c 'unchanged 'gray 'gray-alpha
                                          'rgb 'rgba)
                               'unchanged]
                       [#:device device (or/c #f device/c) #f])
         tensor?]{
Decodes the encoded image @racket[bs], which must not be empty, on the
CPU and moves it to @racket[device], or else to the default device.
@racket['unchanged] keeps the channels the file stores, except that a
palette PNG comes back as RGB, or RGBA with a transparency chunk; the
other modes convert to one, two, three or four channels, the way
torchvision's @tt{ImageReadMode} does, gray as stb's integer luma
@tt{(77r + 150g + 29b) >> 8}. What is not a JPEG or a PNG, or is
truncated, is an error naming the reason. So is an image of more than
178,956,970 pixels, Pillow's decompression-bomb limit, which is refused
from its header before any pixel is decoded, so a small file cannot
claim gigabytes of memory.
}

@defproc[(read-image [path path-string?]
                     [#:mode mode (or/c 'unchanged 'gray 'gray-alpha
                                        'rgb 'rgba)
                             'unchanged]
                     [#:device device (or/c #f device/c) #f])
         tensor?]{
@racket[decode-image] on the contents of the file at @racket[path].
}

@section{Pretrained weights}

@defmodule[torch/vision/weights]

torchvision's ImageNet weights, as the safetensors files timm publishes
on Hugging Face (@tt{timm/resnet18.tv_in1k} and its two siblings), each
pinned to one commit; @filepath{scripts/check-weights.py} confirms they
hold torchvision's tensors, bit for bit. A checkpoint is fetched the first
time it is asked for, checked against the size and SHA-256 recorded in
the module before it reaches the cache, and read from the cache after
that. A @filepath{.txt} file beside it records where the weights came
from and their licence, torchvision's BSD-3-Clause.
@envvar{RKTORCH_WEIGHTS_DIR} moves the cache and
@envvar{RKTORCH_WEIGHTS_URL} points at a Hugging Face mirror, which keeps
the hub's layout, @tt{REPO/resolve/REVISION/model.safetensors}. The files
keep torchvision's key names, and a model loads one with
@racket[load-state!]'s @racket[#:rename].

@defthing[pretrained-weights-names (listof symbol?)]{
The published checkpoints, torchvision's @tt{IMAGENET1K_V1} weights for
three networks: @racket['resnet18-imagenet1k-v1],
@racket['resnet34-imagenet1k-v1] and @racket['resnet50-imagenet1k-v1].
}

@defproc[(pretrained-weights [name symbol?]) path?]{
The path of the cached checkpoint @racket[name], fetching it first when
the cache lacks it. A download whose size or checksum differs from the
published file is an error, and nothing is kept.
}

@defproc[(pretrained-weights-cached? [name symbol?]) boolean?]{
Whether @racket[name] is already in the cache, so a caller can decide
whether to trigger the fetch; the tests only touch cached weights.
}

@section{Images}

@defmodule[torch/vision/ppm]

Sample grids as image files without a dependency: the binary PPM format
(@tt{P6}) is a one-line header followed by one byte per channel, which every
image viewer and converter reads.

@racketblock[
(write-ppm "samples.ppm" (image-grid samples #:columns 10) #:range '(-1 1))
]

@defproc[(image-grid [images tensor?]
                     [#:columns columns exact-positive-integer? 8]
                     [#:padding padding exact-nonnegative-integer? 2]
                     [#:pad-value pad-value (fill-value/c (tensor-dtype images)) 0])
         tensor?]{
Lays the @tt{[N C H W]} batch @racket[images], a rank-4 tensor with at
least one image and no zero dimension, out as one @tt{[C H' W']} image,
@racket[columns] across
and @racket[padding] pixels of @racket[pad-value] around every image, on
the device the batch lives on. One channel becomes three. A batch of one
image comes back as that image, with no border, which is what
@tt{make_grid} returns there. The layout is torchvision's
@tt{make_grid}. @racket[pad-value] must be a value the batch's dtype
holds exactly, so a uint8 batch takes 0 through 255. The grid is built
under @racket[with-no-grad], as @tt{make_grid} is decorated with
@tt{no_grad}: it is a picture of the batch, not a step in its graph. The one-image case returns the batch's own image, so there it
carries whatever the batch carried, again as @tt{make_grid} does.
}

@defproc[(write-ppm [path path-string?]
                    [image tensor?]
                    [#:range range (list/c real? real?) '(0 1)])
         void?]{
Writes the @tt{[3 H W]} tensor @racket[image], whose @tt{H} and @tt{W}
are the positive dimensions its header states, to @racket[path]. A float
image is quantized the way torchvision's @tt{save_image} does, with
@racket[range] naming the values that map to 0 and 255, its first below
its second and both finite, so a dataset in @tt{[-1, 1]} passes
@racket['(-1 1)]; a uint8
image is written as it is. @racket[image] is a float tensor, one of
@racket['float16], @racket['bfloat16], @racket['float32] and
@racket['float64], or a @racket['uint8] one: an integer or boolean image
has no range the transform can read, and under the default
@racket[range] a 0-to-255 integer image would quantize to white rather
than to itself.
}
