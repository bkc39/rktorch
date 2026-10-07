#lang scribble/manual
@(require "../common.rkt"
          (for-label (except-in racket/base
                                abs cos exp log sin sort sqrt max min length
                                + - * /)
                     torch
                     torch/nn))

@title[#:tag "guide-transformers"]{Attention and transformers}

A transformer reads a sequence by letting every position look at every
other and take a weighted average of what it finds. That step is
@deftech{attention}, and the rest of the architecture is arranged around
it. This chapter computes it by hand on tensors small enough to read, then
hands the same work to the library's fused call.

@section[#:tag "transformers-by-hand"]{Attention by hand}

Attention takes three tensors. Each row of the @emph{query} asks a
question, each row of the @emph{key} is what a position offers to be
matched against, and each row of the @emph{value} is what that position
hands back when it matches. Three queries, four keys and four values, all
two wide:

@torch-examples[
(require torch)
(define query (tensor '((1.0 0.0) (0.0 1.0) (1.0 1.0))))
(define key (tensor '((1.0 0.0) (0.0 1.0) (-1.0 0.0) (0.0 -1.0))))
(define value (tensor '((1.0 2.0) (3.0 4.0) (5.0 6.0) (7.0 8.0))))
]

A query matches a key by their dot product, so one matrix product scores
every query against every key, one row per query and one column per key:

@torch-examples[
(define scores (|@| query (T key)))
scores
]

The scores are divided by the square root of the width, two here, so that
wider keys do not make the softmax that follows sharper. The softmax along
each row turns a query's scores into weights that sum to one:

@torch-examples[
(define weights (softmax (/ scores (sqrt 2)) -1))
weights
]

The first query matched the first key best and gives it the largest
weight. Each query's answer is its weights applied to the values:

@torch-examples[
(define by-hand (|@| weights value))
by-hand
]

@racket[scaled-dot-product-attention] is those four steps as one call,
which on a GPU runs as a single fused kernel:

@torch-examples[
(define fused (scaled-dot-product-attention query key value))
fused
(~> (- fused by-hand) abs max item (< 1e-6))
]

@section[#:tag "transformers-causal"]{Hiding the future}

When a sequence attends to itself the queries, keys and values all come
from the same rows. A model that predicts the next token must not read
it, so each position may attend only to itself and the positions before
it. By hand, that is a mask over the scores: fill every later position
with @racket[-inf.0], which the softmax turns into a weight of zero.

@torch-examples[
(define x (tensor '((1.0 0.0) (0.0 1.0) (1.0 1.0))))
(define later (~> (ones 3 3) tril (eq 0)))
later
(define causal-weights
  (~> (|@| x (T x)) (/ (sqrt 2)) (masked-fill later -inf.0) (softmax -1)))
causal-weights
]

The first position sees only itself, so its whole weight sits there. The
fused call takes the same restriction as @racket[#:causal? #t]:

@torch-examples[
(define causal (scaled-dot-product-attention x x x #:causal? #t))
causal
(~> (- causal (|@| causal-weights x)) abs max item (< 1e-6))
]

A boolean @racket[#:mask] states the restriction position by position. Its
sense is the opposite of @racket[later] above: @racket[#t] marks a key the
query may attend to, so the causal mask is the lower triangle itself.

@torch-examples[
(define attend (tril (ones 3 3 #:dtype 'bool)))
(~> (- causal (scaled-dot-product-attention x x x #:mask attend))
    abs max item (< 1e-6))
]

The two senses are easy to confuse, and PyTorch itself uses both;
@secref["attention-function"] in the reference lays out which form takes
which, along with float masks that add to the scores. Everything else in
a transformer is arrangement around this one call: several attentions side
by side, a feed-forward layer after them, and a record of where each token
sits.
