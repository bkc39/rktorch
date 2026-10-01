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
(< (item (max (abs (- fused by-hand)))) 1e-6)
]

@section[#:tag "transformers-causal"]{Hiding the future}

When a sequence attends to itself the queries, keys and values all come
from the same rows. A model that predicts the next token must not read
it, so each position may attend only to itself and the positions before
it. By hand, that is a mask over the scores: fill every later position
with @racket[-inf.0], which the softmax turns into a weight of zero.

@torch-examples[
(define x (tensor '((1.0 0.0) (0.0 1.0) (1.0 1.0))))
(define later (eq (tril (ones 3 3)) 0))
later
(define causal-weights
  (softmax (masked-fill (/ (|@| x (T x)) (sqrt 2)) later -inf.0) -1))
causal-weights
]

The first position sees only itself, so its whole weight sits there. The
fused call takes the same restriction as @racket[#:causal? #t]:

@torch-examples[
(define causal (scaled-dot-product-attention x x x #:causal? #t))
causal
(< (item (max (abs (- causal (|@| causal-weights x))))) 1e-6)
]

A boolean @racket[#:mask] states the restriction position by position. Its
sense is the opposite of @racket[later] above: @racket[#t] marks a key the
query may attend to, so the causal mask is the lower triangle itself.

@torch-examples[
(define attend (tril (ones 3 3 #:dtype 'bool)))
(< (item (max (abs (- causal
                      (scaled-dot-product-attention x x x #:mask attend)))))
   1e-6)
]

The two senses are easy to confuse, and PyTorch itself uses both;
@secref["attention-function"] in the reference lays out which form takes
which, along with float masks that add to the scores. Everything else in
a transformer is arrangement around this one call, starting with several
attentions side by side.

@section[#:tag "transformers-heads"]{Heads and masks}

One attention compares a query with a key along their whole width at once,
so it can weigh the positions in only one way. Multi-head attention
projects the queries, keys and values, cuts each projection into
@deftech{heads}, narrower slices that attend independently, and joins the
heads' answers back together. The layer is @racket[MultiheadAttention].
Here it is on a toy batch, two sequences of three tokens each four wide,
split into two heads two wide:

@torch-examples[
(require torch/nn)
(manual-seed! 0)
(define mha (MultiheadAttention 4 #:heads 2 #:batch-first? #t))
(map car (named-children mha))
(define tokens (randn 2 3 4))
(shape (mha tokens tokens tokens))
]

@racket[#:batch-first? #t] takes the batch as the first axis, as the toy
batch has it. Without it, the layer expects the sequence first, as
PyTorch's does.

The four children are @racket[Linear] layers. The first three project the
input, and the heads are nothing more than a reshape of those projections:
the four columns become two heads of two, and moving the head axis in
front of the tokens makes each head a batch of its own for
@racket[scaled-dot-product-attention]. The last child projects the joined
heads back. By hand:

@torch-examples[
(define (heads t)
  (transpose (reshape t 2 3 2 2) 1 2))
(define per-head
  (scaled-dot-product-attention (heads ((child-ref mha "query") tokens))
                                (heads ((child-ref mha "key") tokens))
                                (heads ((child-ref mha "value") tokens))))
(shape per-head)
(define joined (reshape (transpose per-head 1 2) 2 3 4))
(< (item (max (abs (- ((child-ref mha "out") joined)
                      (mha tokens tokens tokens)))))
   1e-6)
]

Sequences in a batch rarely have the same length. The shorter ones are
padded at the end, and a @deftech{padding mask} keeps every query from
attending to the padding. Say the second sequence is only two tokens
long. Its padding mask is @racket[#t] at the third position, and asking for
the weights shows that no query looks there:

@torch-examples[
(define padding (eq (tensor '((0 0 0) (0 0 1))) 1))
(define-values (out weights)
  (mha tokens tokens tokens #:key-padding-mask padding #:need-weights? #t))
weights
]

Mind the sense: this mask is @racket[#t] where a key is @emph{hidden}, the
opposite of the boolean mask @racket[scaled-dot-product-attention] took
above. The layer mirrors PyTorch's @tt{nn.MultiheadAttention}, which
reads its masks that way, and turns them around itself before the fused
call. The full table is in @secref["attention-multihead-apply"].

The weights come back averaged over the heads, one row per query, each
summing to one. @racket[#:average-attn-weights? #f] keeps the heads apart,
and leaving out @racket[#:need-weights?] skips the weights altogether and
lets the fused kernel do the work, which is how a model runs the layer:
ask for the weights only to look at them.

@racket[#:causal? #t] hides the future, as it did for the single attention
above, and it combines with the padding mask:

@torch-examples[
(define-values (causal-out causal-weights)
  (mha tokens tokens tokens #:causal? #t #:key-padding-mask padding
       #:need-weights? #t))
causal-weights
]

Every query so far came from the same sequence as the keys:
@deftech{self-attention}, one tensor passed three times. In
@deftech{cross-attention} the queries come from one sequence and the keys
and values from another, as when a translation decoder reads the
encoded source sentence. The two can differ in length, and in width too
when @racket[#:key-dim] and @racket[#:value-dim] say so. Three decoder
positions, four wide, reading five encoded ones six wide:

@torch-examples[
(define reader (MultiheadAttention 4 #:heads 2 #:batch-first? #t
                                   #:key-dim 6 #:value-dim 6))
(define encoded (randn 2 5 6))
(define-values (attended attended-weights)
  (reader (randn 2 3 4) encoded encoded #:need-weights? #t))
(shape attended)
(shape attended-weights)
]

The answer has one row per query, in the queries' width; the weights have
one row per query and one column per encoded position.

That is the whole of multi-head attention: projections, heads cut from
them, one fused attention over every head at once, and a projection back.
A transformer stacks it with a feed-forward layer and residual
connections; @secref["ex-gpt"] builds one end to end.
