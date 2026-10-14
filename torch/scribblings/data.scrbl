#lang scribble/manual

@(require (for-label (except-in racket/base length)
                     racket/contract
                     racket/sequence
                     (only-in torch
                              device? draw-seed generator? index-select length
                              make-generator narrow randn randperm seed/c select
                              size/c stack tensor tensor?)
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

@defform[(define-dataset name (field ...) clause ...)
         #:grammar
         ([field id
                 [id default-expr]
                 (code:line keyword id)
                 (code:line keyword [id default-expr])]
          [clause (code:line #:init (formal ...) init-body ...)
                  (code:line #:init (formal ... #:rest rest-id) init-body ...)
                  (code:line #:init (formal ... . rest-id) init-body ...)
                  (code:line #:length length-expr)
                  (code:line #:ref (index-id) body ...+)
                  (code:line #:batch (indices-id collate-id) body ...+)
                  (code:line #:device device-expr)
                  (code:line #:contract contract-expr)
                  (code:line #:predicate id)]
          [formal id
                  [id default-expr]
                  (code:line keyword id)
                  (code:line keyword [id default-expr])])
         #:contracts ([contract-expr contract?])]{

Defines a map-style dataset: a constructor @racket[name], a predicate
@racket[name?], and a structure with one slot per @racket[field] that
implements @racket[gen:dataset]. The clauses are the methods of a
@tt{Dataset} subclass, with every field in scope: @racket[#:init] is
@tt{__init__}, @racket[#:length] is @tt{__len__}, and @racket[#:ref] is
@tt{__getitem__}, returning the item at @racket[index-id] as one value per
field. Both are required.

@racketblock[
(define-dataset squares (n)
  #:init (n)
  #:length n
  #:ref (i) (values (full (* i i) 2) (tensor i)))
]

@racket[#:init]'s @racket[formal]s are the constructor's arguments, in the
grammar of @racket[define]. Every field starts as @racket[#f], or as the
argument of the same name when a formal shares it, and @racket[init-body]
assigns fields with @racket[set!]. Without @racket[#:init], the fields are
themselves the constructor formals.

@racket[#:batch] replaces the default batch, which fetches each item with
@racket[#:ref] and hands the list of items to @racket[collate-id]; use it
when a whole batch is one native op, as @racket[tensor-dataset] does.
@racket[indices-id] is an @racket[indices/c], a list or an int64 tensor;
@racket[indices->list] reads either. @racket[#:device] answers
@racket[dataset-device], @racket[#f] by default; it needs @racket[#:batch],
because a loader then hands @racket[#:batch] an index tensor resident on
that device, which the default batch could only read back with a copy
per batch.

@racket[#:contract] provides the constructor under @racket[contract-expr]
and the predicate under its lowercase name, or the @racket[#:predicate]
one, as @racket[define-layer] does; without it nothing is exported.
}

@defthing[gen:dataset any/c]{
The generic interface a @racket[define-dataset] structure implements, with
methods @racket[dataset-length], @racket[dataset-ref],
@racket[dataset-batch], and @racket[dataset-device]. A structure written
by hand implementing the first two gets the other two by default.
}

@defproc[(indices->list [indices indices/c])
         (non-empty-listof exact-nonnegative-integer?)]{
The indices of a batch as a list, whether a loader handed them over as a
list or as an int64 tensor.
}

@defthing[indices/c contract?]{
A non-empty list of natural numbers, or a non-empty rank-one int64 tensor.
The tensor's elements are not read: a loader cuts them from a permutation,
possibly on a device, and reading them back would wait on it every batch.
}

@defthing[collate/c contract?]{
A procedure from a non-empty list of items, each a list of fields, to the
batch's values.
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
The number of items; @racket[length] answers the same for a dataset
written with @racket[define-dataset].
}

@defproc[(dataset-ref [ds dataset?] [i exact-nonnegative-integer?]) any]{
The item at @racket[i], below the length, as one value per field, as
@tt{dataset[i]} in PyTorch returns a tuple.
}

@defproc[(dataset-batch [ds dataset?]
                        [indices indices/c]
                        [collate collate/c])
         any]{
The batch at @racket[indices], as values. @racket[indices] is a non-empty
list of natural numbers below the length, or a non-empty rank-one int64
tensor when a loader cuts it from a permutation.
}

@defproc[(tensor-dataset [t tensor?] [more tensor?] ...) tensor-dataset?]{
A dataset over one or more tensors of rank at least one sharing their
first dimension and their device, as @tt{TensorDataset}, written with
@racket[define-dataset]: item @racket[i] is @racket[(select t 0 i)] per
tensor. Anything else is a contract violation blamed on the caller.
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
values. Every item must carry the same fields, each of one shape on one
device.
}

@defproc[(default-collate? [v any/c]) boolean?]{
Recognises @racket[default-collate] however it arrived, through any number
of contract boundaries. A @racket[#:batch] body that computes the default
batch natively asks this before taking its fast path, and hands the items
to any other collate.
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
built on the calling thread when they are asked for. A shuffled loader
needs a non-empty dataset within @racket[size/c], as @tt{RandomSampler}
and @racket[randperm] do; an empty one may still be traversed in order. Every traversal
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
The number of batches in a traversal, @tt{len(loader)}; @racket[length]
answers the same.
}

@defproc[(in-dataloader [loader dataloader?]) sequence?]{
One epoch: a sequence of the batches, each as the values its collate
returns, for the @racket[for] forms. A second traversal is a second epoch.
A loader is itself a sequence, so @racket[(for ([(xb yb) loader]) ...)]
is the same epoch, as @tt{for xb, yb in loader} is.
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

@defthing[size/c flat-contract?]{
A natural number below @racket[(expt 2 63)], the native size range.
}

@defproc[(randperm [n size/c]
                   [#:generator generator (or/c generator? #f) #f])
         tensor?]{
An int64 permutation of @racket[0] to @racket[n-1] on the CPU, drawn from
@racket[generator] or from the global stream, as @tt{torch.randperm}. Drawn
on the CPU whatever the default device, as PyTorch's sampler does, so a
permutation replays across devices.
}

@defproc[(draw-seed [#:generator generator (or/c generator? #f) #f])
         exact-nonnegative-integer?]{
One int64 drawn from @racket[generator] or the global stream, as
@tt{torch.empty((), dtype=torch.int64).random_(generator=g).item()}. A
@tt{DataLoader} makes this draw once per epoch before its permutation;
@racket[dataloader] makes it too, so the streams stay in step.
}
