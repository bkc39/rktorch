#lang scribble/manual

@(require "common.rkt"
          torch/scribblings/labels)

@title[#:tag "metrics"]{Metrics}

@defmodule[torch/nn #:link-target? #f]

Classification and language-model figures over a batch, and accumulators
that add batches up to an epoch's figure. Each answers a tensor on its
inputs' device, so a validation step can @racket[log!] it without a read to
the host.

@defproc[(accuracy [predictions tensor?]
                   [labels tensor?]
                   [#:top-k top-k exact-positive-integer? 1]
                   [#:ignore-index ignore-index (or/c #f exact-integer?) #f]
                   [#:validate-args? validate-args? boolean? #t])
         tensor?]{
The fraction of rows whose label is among the @racket[top-k] largest
entries of its row of @racket[predictions], as a @racket['float32] scalar.
@racket[predictions] are @tt{[N, C]} floating-point scores, or
@tt{[N]} @racket['int64] class indices (then @racket[top-k] is 1).
@racket[labels] are @tt{[N]} @racket['int64]. At @racket[top-k] 1 the
prediction is @racket[argmax]'s, the first of equal scores. Rows labelled
@racket[ignore-index] count in neither the numerator nor the denominator;
with none left, the answer is @racket[+nan.0]. @racket[validate-args?]
checks every label is in range by reading their extremes to the host; pass
@racket[#f] in a loop whose labels are known good. This is torchmetrics'
@tt{multiclass_accuracy} with @tt{average="micro"}.

@racketblock[
(define logits (tensor '((2.0 1.0 0.1) (0.5 2.5 0.3) (0.2 0.1 3.0) (1.5 1.6 0.0))))
(define labels (tensor '(0 1 2 0)))
(accuracy logits labels)
(accuracy logits labels #:top-k 2)]
}

@defproc[(confusion-matrix [predictions tensor?]
                           [labels tensor?]
                           [#:classes classes exact-positive-integer?]
                           [#:ignore-index ignore-index (or/c #f exact-integer?) #f]
                           [#:validate-args? validate-args? boolean? #t])
         tensor?]{
The @racket['int64] @tt{[classes, classes]} matrix whose entry
(i, j) counts the rows labelled i and predicted j. Rows are true classes
and columns predicted ones, as in torchmetrics'
@tt{multiclass_confusion_matrix}. Scores are reduced by @racket[argmax]
first. Its diagonal over its sum is the top-1 @racket[accuracy].

@racketblock[
(confusion-matrix (tensor '(0 2 1 1)) (tensor '(0 1 1 2)) #:classes 3)]
}

@defproc[(perplexity [nll (or/c real? tensor?)]
                     [#:tokens tokens (or/c #f exact-positive-integer? tensor?) #f])
         (or/c tensor? flonum?)]{
The exponential of the mean negative log-likelihood per token: @racket[nll]
summed over @racket[tokens] tokens, which is what a strided evaluation
accumulates, or a mean cross-entropy when @racket[tokens] is @racket[#f].
A real answers a flonum; a tensor answers a scalar on its device, and
gradients flow through it.

@racketblock[
(perplexity (cross-entropy (zeros 4 10) (tensor '(0 1 2 3))))
(perplexity 23.0 #:tokens 10)]
}

@deftogether[(@defproc[(accuracy-metric
                        [#:top-k top-k exact-positive-integer? 1]
                        [#:ignore-index ignore-index (or/c #f exact-integer?) #f]
                        [#:validate-args? validate-args? boolean? #t])
                       metric?]
              @defproc[(confusion-matrix-metric
                        [#:classes classes exact-positive-integer?]
                        [#:ignore-index ignore-index (or/c #f exact-integer?) #f]
                        [#:validate-args? validate-args? boolean? #t])
                       metric?]
              @defproc[(perplexity-metric
                        [#:ignore-index ignore-index (or/c #f exact-integer?) #f]
                        [#:validate-args? validate-args? boolean? #t])
                       metric?])]{
Accumulators for @racket[metric-update!]; their keywords mean what they
mean for @racket[accuracy] and @racket[confusion-matrix].
@racket[perplexity-metric] takes logits @tt{[N, V]} or
@tt{[B, T, V]} with targets @tt{[N]} or @tt{[B, T]}, and adds
their summed negative log-likelihood and token count. The counts live on
the device of the first update.
}

@defproc[(metric-update! [m metric?] [predictions tensor?] [targets tensor?]) void?]{
Adds one batch, with the arguments the metric's functional form takes. It
reads nothing to the host unless the metric validates its arguments, and
builds no graph.

@racketblock[
(define correct (accuracy-metric))
(metric-update! correct (narrow logits 0 0 2) (narrow labels 0 0 2))
(metric-update! correct (narrow logits 0 2 2) (narrow labels 0 2 2))
(metric-compute correct)]
}

@defproc[(metric-compute [m metric?]) tensor?]{
The figure over every update since construction or the last
@racket[metric-reset!]: the functional form applied to all of them at
once. It is a fresh tensor, so later updates leave it alone.
}

@defproc[(metric-reset! [m metric?]) void?]{
Forgets every update and the device.
}

@defproc[(metric? [v any/c]) boolean?]{
Whether @racket[v] is an accumulator.
}
