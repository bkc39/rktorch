#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     racket/sequence
                     (only-in torch
                              device? draw-seed generator? index-select seed/c
                              make-generator narrow randn randperm select
                              stack tensor tensor?)
                     torch/data/loader))

@title{Datasets and loaders}

@defmodule[torch/data/loader]

The data pipeline is PyTorch's, without the class hierarchy: a
@deftech{dataset} answers a length and an item per index, a
@deftech{loader} cuts a sequence of indices into batches, and a
@deftech{collate} turns a batch of items into the tensors a training step
takes. One traversal of a loader is one epoch.

@racketblock[
(define loader
  (dataloader (tensor-dataset xs ys)
              #:batch-size 64 #:shuffle? #t
              #:generator (make-generator 0)))
(for* ([epoch (in-range 3)]
       [(xb yb) (in-dataloader loader)])
  (train-step! xb yb))
]

@section{Datasets}

@defthing[gen:dataset any/c]{
The generic interface of a map-style dataset, with methods
@racket[dataset-length], @racket[dataset-ref], @racket[dataset-batch], and
@racket[dataset-device]. A structure implementing the first two gets
@racket[dataset-batch] by default: the items at the indices, each as a
list of its fields, handed to the collate. A dataset over tensors overrides
it with one native op per batch.
}

@defproc[(dataset-device [ds dataset?]) (or/c device? #f)]{
Where the dataset's batches live, or @racket[#f] when it has no single
device. A loader moves each epoch's permutation there once, so no batch
waits on a host-to-device copy. The default is @racket[#f]; a
@racket[tensor-dataset] answers its first tensor's device.
}

@defproc[(dataset? [v any/c]) boolean?]{
Recognises values implementing @racket[gen:dataset].
}

@defproc[(dataset-length [ds dataset?]) exact-nonnegative-integer?]{
The number of items.
}

@defproc[(dataset-ref [ds dataset?] [i exact-nonnegative-integer?]) any]{
The item at @racket[i], as one value per field, as
@tt{dataset[i]} in PyTorch returns a tuple.
}

@defproc[(dataset-batch [ds dataset?]
                        [indices (or/c (listof exact-nonnegative-integer?)
                                       tensor?)]
                        [collate (-> (non-empty-listof list?) any)])
         any]{
The batch at @racket[indices], as values. @racket[indices] is a non-empty
list of natural numbers, or a non-empty rank-one int64 tensor when a
loader cuts it from a permutation.
}

@defproc[(tensor-dataset [t tensor?] [more tensor?] ...) dataset?]{
A dataset over one or more tensors of rank at least one sharing their
first dimension and their device, as @tt{TensorDataset}: item @racket[i]
is @racket[(select t 0 i)] per tensor. Anything else is a contract
violation blamed on the caller.
With @racket[default-collate], its batches never go through items: a
contiguous ascending run of indices is a @racket[narrow] of each tensor,
and any other run is one @racket[index-select] with the indices on the
tensor's device. Keep the tensors on the training device and a batch costs
no host copy. A custom collate sees the items, as @tt{collate_fn} does.
}

@defproc[(tensor-dataset? [v any/c]) boolean?]{
Recognises the result of @racket[tensor-dataset].
}

@defproc[(default-collate [items (non-empty-listof (non-empty-listof tensor?))])
         any]{
@tt{default_collate} for tensor fields: one @racket[stack] per field, as
values. Every item must carry the same number of fields.
}

@section{Loaders}

@defproc[(dataloader [ds dataset?]
                     [#:batch-size batch-size exact-positive-integer? 1]
                     [#:shuffle? shuffle? boolean? #f]
                     [#:drop-last? drop-last? boolean? #f]
                     [#:collate collate (-> (non-empty-listof list?) any)
                      default-collate]
                     [#:generator generator (or/c generator? #f) #f])
         dataloader?]{
A loader over @racket[ds], as @tt{DataLoader(ds, batch_size, shuffle,
drop_last, collate_fn, generator)} with @tt{num_workers=0}: batches are
built on the calling thread when they are asked for. Every traversal
draws what one @tt{DataLoader} iterator draws, in its order, from
@racket[generator] or else the global stream: one @racket[draw-seed] when
the traversal starts, shuffled or not; with @racket[#:shuffle?] the
permutation when the first batch is asked for, and once it is used up the trailing permutation its sampler
discards, before a final partial batch or else when the traversal is
exhausted. A traversal abandoned early leaves the stream where PyTorch's
would. The stream continues across traversals, so
@racket[(make-generator s)] here and
@tt{torch.Generator().manual_seed(s)} there yield the same batch order
epoch after epoch. Without a generator the draws come from the global
stream the way @tt{RandomSampler} makes them, seeding a fresh generator
per epoch. Without @racket[#:shuffle?] the batches are the items in order,
and a batch size equal to the length yields the dataset's own tensors.
}

@defproc[(dataloader? [v any/c]) boolean?]{
Recognises the result of @racket[dataloader].
}

@defproc[(dataloader-length [loader dataloader?]) exact-nonnegative-integer?]{
The number of batches in a traversal, @tt{len(loader)}.
}

@defproc[(in-dataloader [loader dataloader?]) sequence?]{
One epoch: a sequence of the batches, each as the values its collate
returns, for the @racket[for] forms. A second traversal is a second epoch.
}

@defproc[(in-epochs [loader dataloader?] [n exact-nonnegative-integer?])
         sequence?]{
@racket[n] epochs in one sequence: each element is the epoch number
followed by the batch's values, so @racket[(for ([(epoch xb yb) (in-epochs
loader n)]) ...)] is the nested @racket[for*] over @racket[in-range] and
@racket[in-dataloader] written as one clause.
}

@section{Generators}

@defmodule[torch #:link-target? #f]

@defproc[(make-generator [seed seed/c]) generator?]{
A CPU random generator with its own stream, @tt{torch.Generator().manual_seed(seed)}.
The seed is a natural number below @racket[(expt 2 64)], the range the
native generator takes.
Draws from it leave the global stream that @racket[randn] and model
initialisation use untouched, so a seeded shuffle does not perturb seeded
parity elsewhere.
}

@defproc[(generator? [v any/c]) boolean?]{
Recognises the result of @racket[make-generator].
}

@defthing[seed/c flat-contract?]{
A natural number below @racket[(expt 2 64)].
}

@defproc[(randperm [n exact-nonnegative-integer?]
                   [#:generator generator generator? #f])
         tensor?]{
An int64 permutation of @racket[0] to @racket[n-1] on the CPU, drawn from
@racket[generator] or from the global stream, as @tt{torch.randperm}. Drawn
on the CPU whatever the default device, as PyTorch's sampler does, so a
permutation replays across devices.
}

@defproc[(draw-seed [#:generator generator generator? #f])
         exact-nonnegative-integer?]{
One int64 drawn from @racket[generator] or the global stream, as
@tt{torch.empty((), dtype=torch.int64).random_(generator=g).item()}. A
@tt{DataLoader} makes this draw once per epoch before its permutation;
@racket[dataloader] makes it too, so the streams stay in step.
}
