#lang scribble/manual

@(require "common.rkt"
          (for-label (except-in racket/base
                                abs cos exp log sin sort sqrt max min length
                                + - * /)
                     racket/contract
                     torch
                     (only-in torch/nn
                              Dropout LSTM Linear MultiheadAttention eval!
                              load-state! multihead-attention?
                              named-parameters parameters)))

@title[#:tag "attention"]{Attention and transformer layers}

Attention lets every position of a sequence read from every other: each
@deftech{query} scores itself against every @deftech{key}, and answers the
average of the @deftech{values} weighted by the softmax of those scores. A
transformer is that one operation arranged into heads and blocks.

@section[#:tag "attention-function"]{Scaled dot-product attention}

@defmodule[torch #:link-target? #f]

@defproc[(scaled-dot-product-attention
          [query tensor?]
          [key tensor?]
          [value tensor?]
          [#:mask mask (or/c tensor? #f) #f]
          [#:causal? causal? boolean? #f]
          [#:dropout dropout (and/c real? (>=/c 0) (</c 1)) 0.0]
          [#:scale scale (or/c real? #f) #f])
         tensor?]{
Attention in one fused call, PyTorch's
@tt{torch.nn.functional.scaled_dot_product_attention}:
@tt{softmax(query @"·" key@superscript{T} @"×" scale + mask) @"·" value},
the softmax taken over the keys.

The shapes are @tt{[..., L, E]} for @racket[query], @tt{[..., S, E]} for
@racket[key] and @tt{[..., S, Ev]} for @racket[value]: @tt{L} queries and
@tt{S} keys of width @tt{E}, and one value of width @tt{Ev} per key. The
answer is @tt{[..., L, Ev]}, one weighted value per query. The leading
dimensions, a batch and usually a head axis, broadcast as @racket[matmul]'s
do. A tensor of rank below two, a @racket[query] and @racket[key] of
different widths, and a @racket[key] and @racket[value] of different
lengths are contract violations.

@racket[scale] multiplies the scores, and defaults to @tt{1/@"√"E}, so
the scores keep unit variance however wide the keys are.

@torch-examples[
(define q (tensor '((1.0 0.0) (0.0 1.0))))
(define v (tensor '((1.0 2.0) (3.0 4.0))))
(scaled-dot-product-attention q q v)
(scaled-dot-product-attention q q v #:scale 0)
]

@racket[mask] says which keys each query may attend to, and broadcasts to
@tt{[..., L, S]}:

@tabular[#:sep @hspace[2]
         #:style 'boxed
         (list (list @bold{@racket[mask]} @bold{effect on the scores})
               (list @racket[#f] "none: every query attends to every key")
               (list @elem{a @racket['bool] tensor}
                     @elem{@racket[#t] where the query may attend, @racket[#f]
                           where the key is hidden})
               (list @elem{a float tensor}
                     @elem{added to the scores before the softmax;
                           @racket[-inf.0] hides a key}))]

The boolean sense is the one PyTorch gives this function: @bold{@racket[#t]
means attend}. PyTorch is not consistent about it.
@tt{nn.MultiheadAttention}'s boolean @tt{attn_mask} and
@tt{key_padding_mask} mean the opposite, @bold{@racket[#t] means hidden},
and every layer here that mirrors @tt{nn.MultiheadAttention} keeps that
sense, as does the @racket[masked-fill] idiom, which fills where its mask
is @racket[#t]. @racket[MultiheadAttention]'s @racket[#:key-padding-mask]
and @racket[#:attn-mask] are such masks, @racket[#t] where a key is
hidden, and the layer turns them around itself before it calls this
function (@secref["attention-multihead-apply"]). A float mask is additive
in both. @racket[(eq mask 0)] turns a boolean mask of one sense into the
other.

@torch-examples[
(define attend (tril (ones 2 2 #:dtype 'bool)))
attend
(scaled-dot-product-attention q q v #:mask attend)
(scaled-dot-product-attention q q v #:mask (eq attend 0))
]

A mask of the wrong sense attends to exactly the keys it meant to hide, as
the last example does. A query left with no key at all, the second row
there, answers zeros rather than the not-a-number a softmax over nothing would
give, on the CPU and on CUDA alike.

@racket[causal?] hides from each query every key after its own position:
query @tt{i} attends to keys @tt{0} through @tt{i}, the mask
@racket[(tril (ones L S #:dtype 'bool))]. It is how a decoder is kept from
reading the tokens it is predicting. A @racket[mask] together with
@racket[causal?] @racket[#t] is a contract violation, as PyTorch refuses
the pair; fold the causal triangle into the mask instead.

@torch-examples[
(scaled-dot-product-attention q q v #:causal? #t)
]

@racket[dropout] is the probability of zeroing each attention weight, the
rest scaled by @tt{1/(1 - dropout)}, drawn from the global random stream.
It applies whenever it is above zero: like PyTorch's function, and unlike
@racket[Dropout], there is no training mode to consult, so a caller that
evaluates passes @racket[0.0] itself.

Each device chooses its own kernel. On CUDA that is a fused one wherever
the inputs allow it, FlashAttention for @racket['float16] and
@racket['bfloat16], otherwise the memory-efficient kernel or cuDNN's,
and the computation above as the fallback. There is nothing to select
here, and the kernels agree up to rounding.
Gradients flow to @racket[query], @racket[key] and @racket[value].

PyTorch's @tt{enable_gqa}, grouped-query attention with fewer key and
value heads than query heads, is not exposed: it stays off.
}

@section[#:tag "attention-multihead"]{Multi-head attention}

@defmodule[torch/nn #:link-target? #f]

@defproc[(MultiheadAttention
          [embed-dim exact-positive-integer?]
          [#:heads heads exact-positive-integer?]
          [#:dropout dropout (and/c real? (>=/c 0) (</c 1)) 0.0]
          [#:bias? bias? boolean? #t]
          [#:batch-first? batch-first? boolean? #f]
          [#:key-dim key-dim exact-positive-integer? embed-dim]
          [#:value-dim value-dim exact-positive-integer? embed-dim])
         multihead-attention?]{
PyTorch's @tt{nn.MultiheadAttention}: @racket[heads] attentions side by
side, each over its own @tt{embed-dim/heads}-wide slice of the projected
query, key and value, their answers joined and projected back to
@racket[embed-dim]. @racket[heads] has to divide @racket[embed-dim];
another count is a contract violation. How the layer applies, its masks
and its weights are in @secref["attention-multihead-apply"].

The projections are four @racket[Linear] children, named @racket["query"],
@racket["key"], @racket["value"] and @racket["out"]. PyTorch fuses the
first three into one @tt{in_proj_weight}; keeping them apart is what lets
a name pick out one projection, as a fine-tuning recipe that adapts only
@racket["query"] and @racket["value"] does, and the same four names serve
every checkpoint mapped onto this layer
(@secref["attention-multihead-pytorch"]).

@tabular[#:sep @hspace[2]
         #:style 'boxed
         (list (list @bold{parameter} @bold{shape})
               (list @racket["query.weight"] @tt{[embed-dim, embed-dim]})
               (list @racket["key.weight"] @tt{[embed-dim, key-dim]})
               (list @racket["value.weight"] @tt{[embed-dim, value-dim]})
               (list @racket["out.weight"] @tt{[embed-dim, embed-dim]})
               (list @elem{@racket["query.bias"], @racket["key.bias"],
                           @racket["value.bias"], @racket["out.bias"]}
                     @tt{[embed-dim]}))]

With @racket[bias?] @racket[#f] the four biases are absent, as with
@tt{bias=False}. @racket[key-dim] and @racket[value-dim] are the widths
of the keys and values when they differ from the queries', PyTorch's
@tt{kdim} and @tt{vdim}, as in a decoder attending to an encoder of
another width. @racket[dropout] applies to the attention weights, in
training mode only.

@torch-examples[
(manual-seed! 0)
(define mha (MultiheadAttention 8 #:heads 2))
(map car (named-parameters mha))
(define cross (MultiheadAttention 8 #:heads 2 #:key-dim 4 #:value-dim 6))
(map shape (parameters cross))
]}

@defproc[(multihead-attention? [v any/c]) boolean?]{
Whether @racket[v] was built by @racket[MultiheadAttention].
}

@subsection[#:tag "attention-multihead-apply"]{Applying multi-head attention}

A @racket[MultiheadAttention] layer applies as
@racket[(mha query key value keyword-arg ...)]. Each of the three is a
sequence: @tt{L} queries, and @tt{S} keys with one value each.

@tabular[#:sep @hspace[2]
         #:style 'boxed
         (list (list "" @bold{default}
                     @bold{@racket[#:batch-first?] @racket[#t]}
                     @bold{unbatched})
               (list @racket[query] @tt{[L, N, embed-dim]}
                     @tt{[N, L, embed-dim]} @tt{[L, embed-dim]})
               (list @racket[key] @tt{[S, N, key-dim]}
                     @tt{[N, S, key-dim]} @tt{[S, key-dim]})
               (list @racket[value] @tt{[S, N, value-dim]}
                     @tt{[N, S, value-dim]} @tt{[S, value-dim]})
               (list "answer" @tt{[L, N, embed-dim]}
                     @tt{[N, L, embed-dim]} @tt{[L, embed-dim]}))]

The default layout puts the sequence first, as PyTorch's and
@racket[LSTM]'s do; a layer built with @racket[#:batch-first?]
@racket[#t] takes and answers the batch first. Rank-two inputs are one
unbatched sequence in either layout. Self-attention passes one tensor three
times; cross-attention passes the queries from one sequence and the keys
and values from another, whose length @tt{S} need not be @tt{L}.

@torch-examples[
(define x (randn 5 2 8))
(shape (mha x x x))
(define memory (randn 7 2 4))
(shape (cross x memory (randn 7 2 6)))
]

The application takes these keywords:

@tabular[#:sep @hspace[2]
         #:style 'boxed
         (list (list @bold{keyword} @bold{accepts} @bold{default})
               (list @racket[#:key-padding-mask]
                     @elem{a mask of shape @tt{[N, S]} (@tt{[S]} unbatched),
                           or @racket[#f]}
                     @racket[#f])
               (list @racket[#:attn-mask]
                     @elem{a mask of shape @tt{[L, S]} for every head, or
                           @tt{[N·heads, L, S]} for each sequence and head
                           (@tt{[heads, L, S]} unbatched), or @racket[#f]}
                     @racket[#f])
               (list @racket[#:causal?] @racket[boolean?] @racket[#f])
               (list @racket[#:need-weights?] @racket[boolean?] @racket[#f])
               (list @racket[#:average-attn-weights?] @racket[boolean?]
                     @racket[#t]))]

A mask is a @racket['bool] tensor or a floating-point one. Inputs of
another width than the layer was built for, inputs that disagree on the
batch or on @tt{S}, and a mask of another shape are contract violations
blaming the caller, the message naming the shapes.

@bold{The boolean masks here read the opposite way from
@racket[scaled-dot-product-attention]'s.} They mirror
@tt{nn.MultiheadAttention}'s, where @racket[#t] marks a key the query may
@emph{not} attend to, the sense of a padding mask and of the
@racket[masked-fill] idiom:

@tabular[#:sep @hspace[2]
         #:style 'boxed
         (list (list @bold{mask} @bold{@racket[MultiheadAttention]}
                     @bold{@racket[scaled-dot-product-attention]})
               (list @elem{@racket['bool], @racket[#t]} "the key is hidden"
                     "the key is attended to")
               (list @elem{@racket['bool], @racket[#f]}
                     "the key is attended to" "the key is hidden")
               (list "float" "added to the scores" "added to the scores"))]

@racket[(eq mask 0)] turns one sense into the other. A float mask is
additive in both, @racket[-inf.0] hiding a key. The layer turns its
boolean masks into float ones itself, so nothing reaches
@racket[scaled-dot-product-attention] in the wrong sense.

@racket[#:key-padding-mask] hides keys per sequence, such as the padding
that ends the shorter sequences of a batch; @racket[#:attn-mask] hides
keys per query. @racket[#:causal?] @racket[#t] hides from query @tt{i}
every key after position @tt{i}, the mask
@racket[(eq (tril (ones L S)) 0)]. PyTorch's @tt{is_causal} is only a hint
that @tt{attn_mask} already holds that mask, and refuses to run without
one; here @racket[#:causal?] builds the mask itself. Masks combine: a key
any of them hides stays hidden, and float masks add.

@torch-examples[
(define tokens (randn 3 2 8))
(define padding (eq (tensor '((0 0 0) (0 1 1))) 1))
(define-values (padded-out padded-weights)
  (mha tokens tokens tokens #:key-padding-mask padding #:need-weights? #t))
padded-weights
(define-values (causal-out causal-weights)
  (mha tokens tokens tokens #:causal? #t #:need-weights? #t))
(ref causal-weights 0)
(define later (eq (tril (ones 3 3)) 0))
(< (item (max (abs (- causal-out
                      (mha tokens tokens tokens #:attn-mask later)))))
   1e-6)
]

The second sequence puts all its weight on its first key, the only one its
padding leaves; under @racket[#:causal?] the first query sees only
itself.

With @racket[#:need-weights?] @racket[#t] the application answers two
values, the output and the attention weights: @tt{[N, L, S]}, the average
over the heads, or @tt{[N, heads, L, S]} with
@racket[#:average-attn-weights?] @racket[#f] (@tt{[L, S]} and
@tt{[heads, L, S]} unbatched), each row summing to one over the keys.
Without it, the answer is the output alone, computed by
@racket[scaled-dot-product-attention]'s fused kernel; the weights need the
softmax written out, which is slower and holds an @tt{[N, heads, L, S]}
tensor. PyTorch's @tt{need_weights} defaults to true and answers @tt{None}
in the weights' place when it is false; here the default is the fused path
and a single value. @racket[#:average-attn-weights?] has no effect without
@racket[#:need-weights?]. The two paths answer the same output, except for
a query whose every key is hidden: the fused kernel attends to nothing and
answers @racket["out.bias"], while the written-out softmax answers
not-a-number, as PyTorch's two paths do.

The layer's @racket[#:dropout] zeroes each attention weight with that
probability, in training mode only; @racket[eval!] turns it off. With
@racket[#:need-weights?] the weights answered are the ones after dropout,
as PyTorch's are.

@subsection[#:tag "attention-multihead-pytorch"]{PyTorch's parameters}

The layer draws its initial values as @tt{nn.MultiheadAttention} does, in
the same order: first @racket["out"], as a @racket[Linear] draws itself,
then one xavier-uniform @tt{[3·embed-dim, embed-dim]} in-projection,
split by rows into @racket["query.weight"], @racket["key.weight"] and
@racket["value.weight"]. When @racket[#:key-dim] or @racket[#:value-dim]
differs from the embedding width there are three draws instead, one per
projection, query first. Every bias starts at zero, @racket["out.bias"]
included, though it is drawn first like any @racket[Linear]'s. So
@racket[(manual-seed! s)] followed by
@racket[(MultiheadAttention e #:heads h)] starts from the values a seeded
@tt{nn.MultiheadAttention(e, h)} holds, and leaves the random stream
where PyTorch leaves it.

PyTorch's parameters map onto the four projections by rows of its fused
tensors, @tt{E} being the embedding width:

@tabular[#:sep @hspace[2]
         #:style 'boxed
         (list (list @bold{@tt{nn.MultiheadAttention}} @bold{here})
               (list @tt{in_proj_weight[0:E]} @racket["query.weight"])
               (list @tt{in_proj_weight[E:2E]} @racket["key.weight"])
               (list @tt{in_proj_weight[2E:3E]} @racket["value.weight"])
               (list @elem{@tt{q_proj_weight}, @tt{k_proj_weight},
                           @tt{v_proj_weight} (other widths)}
                     @elem{@racket["query.weight"], @racket["key.weight"],
                           @racket["value.weight"]})
               (list @elem{@tt{in_proj_bias}, in thirds}
                     @elem{@racket["query.bias"], @racket["key.bias"],
                           @racket["value.bias"]})
               (list @elem{@tt{out_proj.weight}, @tt{out_proj.bias}}
                     @elem{@racket["out.weight"], @racket["out.bias"]}))]

@racket[load-state!]'s renaming covers @tt{out_proj}; the fused tensors
need splitting by rows before they load. PyTorch's @tt{add_bias_kv},
@tt{add_zero_attn} and its inference fast path over nested tensors are not
supported.
