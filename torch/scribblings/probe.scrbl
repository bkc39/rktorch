#lang scribble/manual

@(require "common.rkt"
          torch/scribblings/labels
          (for-label (only-in torch/data/loader dataset? tensor-dataset)))

@title[#:tag "probes"]{Probes}

@defmodule[torch/train #:link-target? #f]

How well a frozen encoder's features separate the classes: a linear
classifier and a k-nearest-neighbour vote over cached features.
Standardisation statistics come from the fit split only; a probe's weight
decay is chosen on the select split; and the test split should be scored
once.

@defproc[(probe-dataset [encoder (or/c layer? procedure?)]
                        [dataset dataset?]
                        [#:pool pool probe-pool/c 'cls]
                        [#:forward forward (or/c #f (procedure-arity-includes/c 1)) #f]
                        [#:batch-size batch-size exact-positive-integer? 1024]
                        [#:device device (or/c #f device/c) #f]
                        [#:classes classes (or/c #f exact-positive-integer?) #f])
         probe-dataset?]{
One pass of @racket[encoder] over @racket[dataset] in @racket['eval] mode
under @racket[with-no-grad], in batches of @racket[batch-size]. It keeps
float32 features and @racket['int64] labels on @racket[device] as a dataset
of @racket[(features label)] items. Each batch's first two values are
images and labels; @racket[forward] takes the images to the encoder's
output, by default @racket[(encoder images)]. @racket[pool] reduces tokens
@tt{[N, T, D]}: @racket['cls] is token 0, @racket['mean] is the mean of the
others, @racket['cls+mean] concatenates the two, and @racket[#f] keeps a
@tt{[N, D]} output as it is. Every layer's mode is restored afterwards.
Images go to the encoder's device: its first parameter's, else the
dataset's, else @racket[(default-device)]; @racket[device] defaults to it.
An empty dataset, or a label outside @tt{[0, classes)}, raises
@racket[exn:fail:contract?].

@examples[#:eval train-eval
(define images (tensor-dataset (randn 32 1 2 2) (tensor (for/list ([i 32]) (modulo i 4)))))
(define encoded (probe-dataset (Sequential (Flatten) (Linear 4 8)) images #:pool #f))
(shape (probe-dataset-features encoded))]
}

@defproc[(tensors->probe-dataset [features tensor?]
                                 [labels tensor?]
                                 [#:classes classes (or/c #f exact-positive-integer?) #f])
         probe-dataset?]{
Features already computed, @tt{[N, D]} @racket['float32] or
@racket['float64], with their @tt{[N]} @racket['int64] labels: at least
one row, labels in @tt{[0, classes)}, features and labels on one device.
The features are kept detached.
}

@deftogether[(@defproc[(probe-dataset-features [ds probe-dataset?]) tensor?]
              @defproc[(probe-dataset-labels [ds probe-dataset?]) tensor?]
              @defproc[(probe-dataset-classes [ds probe-dataset?]) exact-positive-integer?]
              @defproc[(probe-dataset-indices [ds probe-dataset?]) (or/c #f tensor?)]
              @defproc[(probe-dataset? [v any/c]) boolean?])]{
The cached tensors; the class count (1 + the largest label unless given);
and the rows of the dataset a split came from, or @racket[#f].
}

@defthing[probe-pool/c flat-contract?]{
Holds for @racket['cls], @racket['mean], @racket['cls+mean] and
@racket[#f].
}

@defproc[(probe-split [ds probe-dataset?]
                      [#:select-per-class per-class exact-positive-integer? 500]
                      [#:seed seed seed/c 0])
         (values probe-dataset? probe-dataset?)]{
A stratified held-out split, the fit split first:
@racket[per-class] rows of each class drawn with a generator seeded
@racket[seed] as the select split, and the rest as the fit split, both in
ascending order. It depends only on the labels and the seed, so features
of the same images split the same way. The defaults give CIFAR-10's
45,000 / 5,000. A class with fewer than @racket[per-class] rows raises
@racket[exn:fail:contract?].
}

@defproc[(linear-probe [fit probe-dataset?]
                       [select (or/c #f probe-dataset?) #f]
                       [#:weight-decays weight-decays (non-empty-listof (>=/c 0))
                                        '(1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1)]
                       [#:weight-decay weight-decay (or/c #f (>=/c 0)) #f]
                       [#:max-iter max-iter exact-positive-integer? 1000])
         linear-probe?]{
Multinomial logistic regression on the fit split's standardised features,
fitted full batch by L-BFGS with strong-Wolfe line search from zero
weights. The objective is mean cross-entropy plus half the weight decay
times the squared weights; the bias is not decayed. With @racket[select],
it fits once per entry of @racket[weight-decays] and keeps the first with
the best select accuracy; its classes must be within the fit's. Otherwise
it fits once at @racket[weight-decay].

@examples[#:eval train-eval
(manual-seed! 0)
(define blobs
  (tensors->probe-dataset (cat (list (+ (randn 60 2) 3.0) (- (randn 60 2) 3.0)))
                          (cat (list (zeros 60 #:dtype 'int64) (ones 60 #:dtype 'int64)))))
(define-values (fit select) (probe-split blobs #:select-per-class 10))
(define probe (linear-probe fit select))
(linear-probe-weight-decay probe)
(probe-accuracy probe select)]
}

@deftogether[(@defproc[(linear-probe-weight [p linear-probe?]) tensor?]
              @defproc[(linear-probe-bias [p linear-probe?]) tensor?]
              @defproc[(linear-probe-weight-decay [p linear-probe?]) (>=/c 0)]
              @defproc[(linear-probe-sweep [p linear-probe?])
                       (or/c #f (listof (cons/c (>=/c 0) (real-in 0 1))))]
              @defproc[(linear-probe-mean [p linear-probe?]) tensor?]
              @defproc[(linear-probe-std [p linear-probe?]) tensor?]
              @defproc[(linear-probe-classes [p linear-probe?]) exact-positive-integer?]
              @defproc[(linear-probe? [v any/c]) boolean?])]{
@tt{[D, C]} weights applied as @racket[x W + b]. The sweep is each weight
decay with its select accuracy, or @racket[#f]. The mean and std are the
standardiser's.
}

@defproc[(probe-accuracy [probe linear-probe?] [ds probe-dataset?]) (real-in 0 1)]{
The probe's top-1 accuracy on @racket[ds], standardised with the probe's
own statistics.
}

@defproc[(knn-accuracy [fit probe-dataset?]
                       [query (or/c #f probe-dataset?) #f]
                       [#:k k exact-positive-integer? 20]
                       [#:temperature temperature (>/c 0) 0.07]
                       [#:standardize? standardize? boolean? #t]
                       [#:batch-size batch-size exact-positive-integer? 1000])
         (real-in 0 1)]{
Top-1 accuracy of a weighted vote among the @racket[k] fit rows most
cosine-similar to each query row, after standardising both with the fit's
statistics when @racket[standardize?], which needs at least two fit rows.
Each neighbour votes @racket[(exp (/ similarity temperature))], scaled
per query row by its nearest neighbour's vote so that a small
@racket[temperature] does not overflow. Without @racket[query] it scores
the fit split, each row leaving itself out.

@examples[#:eval train-eval
(knn-accuracy fit)
(knn-accuracy fit select)]
}
