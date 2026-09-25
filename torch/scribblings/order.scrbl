#lang scribble/manual

@(require (for-label (except-in racket/base sort)
                     racket/contract/base
                     (only-in torch
                              argsort generator? log-softmax make-generator
                              manual-seed! multinomial softmax sort tensor?
                              topk)
                     (only-in torch/nn nll-loss)))

@title{Ordering and sampling}

@defmodule[torch #:link-target? #f]

The operations a decoder needs to turn scores into tokens: the largest
entries, a full ordering, and a draw from a categorical distribution.
@racket[topk] and @racket[sort] return two tensors as two values, the
entries and the @racket['int64] positions they came from.

@defproc[(topk [t tensor?]
               [k exact-nonnegative-integer?]
               [#:dim dim exact-integer? -1]
               [#:largest? largest? boolean? #t]
               [#:sorted? sorted? boolean? #t])
         (values tensor? tensor?)]{
The @racket[k] largest entries of @racket[t] along @racket[dim], or the
smallest when @racket[largest?] is @racket[#f], and their indices along
that dimension, as PyTorch's @tt{torch.topk}. A @racket[k] beyond the
length of @racket[dim] is a contract violation. Gradients flow to the
selected entries. Greedy decoding is @racket[k] = 1:

@racketblock[
(define-values (score token) (topk logits 1))
]
}

@defproc*[([(sort [t tensor?]
                  [#:dim dim exact-integer? -1]
                  [#:descending? descending? boolean? #f])
            (values tensor? tensor?)]
           [(sort [lst list?]
                  [less-than? (-> any/c any/c any/c)]
                  [#:key key (-> any/c any/c) (lambda (x) x)]
                  [#:cache-keys? cache-keys? boolean? #f])
            list?])]{
A tensor is sorted along @racket[dim] as PyTorch's @tt{torch.sort} does,
answering the sorted tensor and the indices that sort it. Anything else
goes to @racketmodname[racket/base]'s @racket[sort] unchanged, so
requiring @racketmodname[torch] leaves list code as it was. The keywords
of one form are a contract violation on the other.
}

@defproc[(argsort [t tensor?]
                  [#:dim dim exact-integer? -1]
                  [#:descending? descending? boolean? #f])
         tensor?]{
The indices half of @racket[sort].
}

@defproc[(multinomial [probabilities tensor?]
                      [num-samples exact-positive-integer?]
                      [#:replacement? replacement? boolean? #f]
                      [#:generator generator (or/c generator? #f) #f])
         tensor?]{
Draws @racket[num-samples] category indices per row of
@racket[probabilities], a vector or a matrix of non-negative weights that
need not sum to one, as PyTorch's @tt{torch.multinomial}. The draw comes
from @racket[generator]'s stream, or the global one seeded by
@racket[manual-seed!]; under a shared seed the indices match PyTorch's on
the CPU. Sampling a token at a temperature:

@racketblock[
(multinomial (softmax (/ logits temperature) -1) 1)
]

An out-of-memory failure here is not retried, since a second attempt
would advance the stream.
}

@section{Negative log likelihood}

@defmodule[torch/nn #:link-target? #f]

@defproc[(nll-loss [log-probs tensor?]
                   [targets tensor?]
                   [#:weight weight (or/c tensor? #f) #f]
                   [#:reduction reduction (or/c 'none 'mean 'sum) 'mean]
                   [#:ignore-index ignore-index exact-integer? -100])
         tensor?]{
The negative log likelihood of integer @racket[targets] under
@racket[log-probs], which are log-probabilities of shape @tt{[N, C]}
(the output of @racket[log-softmax], not raw logits), as PyTorch's
@tt{F.nll_loss}. Targets equal to @racket[ignore-index] contribute
nothing, the way a padded position is masked out of a sequence loss;
@racket[weight] rescales each class.
}
