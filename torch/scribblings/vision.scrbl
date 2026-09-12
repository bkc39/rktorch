#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     (only-in torch device? tensor?)
                     torch/data/loader
                     torch/vision/cifar10))

@title{Vision datasets}

@defmodule[torch/vision/cifar10]

CIFAR-10 from its binary distribution: 60000 colour images of 32 by 32
pixels in ten classes, 50000 for training and 10000 for testing. The
archive is fetched once into the cache directory (or
@envvar{RKTORCH_CIFAR10_DIR}) and unpacked in memory; nothing else is
written.

@racketblock[
(define loader
  (dataloader (cifar10-dataset 'train #:device (cuda-if-available))
              #:batch-size 128 #:shuffle? #t))
]

@defproc[(load-cifar10 [split (or/c 'train 'test) 'train])
         (values tensor? tensor?)]{
The split's images as a float32 tensor of shape @tt{[N 3 32 32]} with
pixels in @tt{[-1, 1]}, and its labels as an int64 tensor of shape
@tt{[N]}, the layout a diffusion model trains on. The five training
batches come back as one tensor.
}

@defproc[(cifar10-dataset [split (or/c 'train 'test) 'train]
                          [#:device device (or/c #f device?) #f])
         dataset?]{
@racket[load-cifar10] as a @racket[tensor-dataset], moved to
@racket[device] when one is given, so a loader over it batches on the
device.
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

@defproc[(cifar10-records->tensors [bs bytes?]) (values tensor? tensor?)]{
Parses a buffer of 3073-byte records, one label byte followed by the
red, green and blue planes of an image, into the tensors above. A buffer
that is not a whole number of records is an error.
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
