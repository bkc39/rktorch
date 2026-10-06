#lang scribble/manual

@(require "common.rkt"
          (for-label (except-in racket/base
                                abs cos exp log sin sort sqrt max min length
                                + - * /)
                     racket/contract
                     torch
                     (only-in torch/nn
                              Dropout Embedding LSTM LayerList LayerNorm Linear
                              MultiheadAttention TransformerDecoder
                              TransformerDecoderLayer TransformerEncoder
                              TransformerEncoderLayer causal-mask child-ref
                              eval! in-layers layer? load-state!
                              multihead-attention? named-children
                              named-parameters parameters sinusoidal-positions
                              transformer-decoder-layer? transformer-decoder?
                              transformer-encoder-layer? transformer-encoder?)
                     (only-in torch/vision/diffusion sinusoidal-embedding)))

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

@section[#:tag "attention-transformer-layers"]{Transformer layers}

@defmodule[torch/nn #:link-target? #f]

@deftogether[(@defproc[(TransformerEncoderLayer
                        [d-model exact-positive-integer?]
                        [#:heads heads exact-positive-integer?]
                        [#:ffn-width ffn-width exact-positive-integer? 2048]
                        [#:dropout dropout (and/c real? (>=/c 0) (</c 1)) 0.1]
                        [#:activation activation
                                      (or/c 'relu 'gelu 'gelu-tanh
                                            (procedure-arity-includes/c 1))
                                      'relu]
                        [#:norm-first? norm-first? boolean? #f]
                        [#:layer-norm-eps layer-norm-eps
                                          (and/c real? positive?) 1e-5]
                        [#:batch-first? batch-first? boolean? #f]
                        [#:bias? bias? boolean? #t])
                       transformer-encoder-layer?]
              @defproc[(TransformerDecoderLayer
                        [d-model exact-positive-integer?]
                        [#:heads heads exact-positive-integer?]
                        [#:ffn-width ffn-width exact-positive-integer? 2048]
                        [#:dropout dropout (and/c real? (>=/c 0) (</c 1)) 0.1]
                        [#:activation activation
                                      (or/c 'relu 'gelu 'gelu-tanh
                                            (procedure-arity-includes/c 1))
                                      'relu]
                        [#:norm-first? norm-first? boolean? #f]
                        [#:layer-norm-eps layer-norm-eps
                                          (and/c real? positive?) 1e-5]
                        [#:batch-first? batch-first? boolean? #f]
                        [#:bias? bias? boolean? #t])
                       transformer-decoder-layer?])]{
PyTorch's @tt{nn.TransformerEncoderLayer} and
@tt{nn.TransformerDecoderLayer}, the two blocks of the original
transformer, over sequences @racket[d-model] wide.

An encoder layer has two sublayers. Self-attention, a
@racket[MultiheadAttention] of @racket[heads] heads, lets every position
read every other; then a feed-forward network reads each position on its
own: a @racket[Linear] out to @racket[ffn-width], the activation, and a
@racket[Linear] back. A decoder layer puts a third sublayer between those
two, a cross-attention whose keys and values are the encoder's output, the
@deftech{memory}. The arguments mean the same in both layers.
@racket[heads] has to divide @racket[d-model]; another count is a contract
violation.

Every sublayer's answer is added to its input, a residual connection, and
a @racket[LayerNorm] keeps the sum in scale. @racket[norm-first?] says
where the normalization goes, written here for one sublayer @tt{f}:

@tabular[#:sep @hspace[2]
         #:style 'boxed
         (list (list @bold{@racket[norm-first?]} @bold{each sublayer}
                     @bold{used by})
               (list @elem{@racket[#f], post-norm}
                     @tt{x ← norm(x + f(x))}
                     "the original transformer, BERT, DistilBERT")
               (list @elem{@racket[#t], pre-norm}
                     @tt{x ← x + f(norm(x))}
                     "GPT-2, ViT"))]

A pre-norm layer never normalizes the sum it passes on, so a stack of them
trains more steadily as it deepens, and ends with a normalization of its
own (@racket[TransformerEncoder]'s @racket[#:norm]).

@racket[activation] is applied between the feed-forward's two
@racket[Linear]s: @racket['relu], PyTorch's default; @racket['gelu], the
exact @racket[gelu]; @racket['gelu-tanh], @racket[gelu]'s tanh
approximation, the one GPT-2 was trained with
(@tt{@literal{F.gelu(x, approximate='tanh')}}); or any procedure from a
tensor to a tensor, as PyTorch accepts a callable.

@racket[dropout] zeroes with that probability, in training mode only, at
four places: the attention weights, each sublayer's answer before it is
added back, and the feed-forward's hidden layer. @racket[eval!] turns all
of them off. @racket[layer-norm-eps] is every @racket[LayerNorm]'s
@racket[#:eps]. @racket[batch-first?] takes and answers the batch first
rather than the sequence, as @racket[MultiheadAttention]'s does.
@racket[bias?] @racket[#f] drops every bias, in the projections, the
feed-forward and the norms, as PyTorch's @tt{bias=False} does.

The children carry PyTorch's attribute names, with an underscore spelled
as a hyphen:

@tabular[#:sep @hspace[2]
         #:style 'boxed
         (list (list @bold{child} @bold{layer} @bold{in PyTorch})
               (list @racket["self-attn"] @racket[MultiheadAttention]
                     @tt{self_attn})
               (list @elem{@racket["multihead-attn"] (decoder)}
                     @elem{@racket[MultiheadAttention] over the memory}
                     @tt{multihead_attn})
               (list @racket["linear1"]
                     @elem{@racket[Linear], @racket[d-model] to
                           @racket[ffn-width]}
                     @tt{linear1})
               (list @racket["dropout"] @racket[Dropout] @tt{dropout})
               (list @racket["linear2"]
                     @elem{@racket[Linear], @racket[ffn-width] to
                           @racket[d-model]}
                     @tt{linear2})
               (list @elem{@racket["norm1"], @racket["norm2"],
                           @racket["norm3"] (decoder)}
                     @racket[LayerNorm]
                     @elem{@tt{norm1}, @tt{norm2}, @tt{norm3}})
               (list @elem{@racket["dropout1"], @racket["dropout2"],
                           @racket["dropout3"] (decoder)}
                     @racket[Dropout]
                     @elem{@tt{dropout1}, @tt{dropout2}, @tt{dropout3}}))]

@torch-examples[
(manual-seed! 0)
(define block (TransformerEncoderLayer 8 #:heads 2 #:ffn-width 32))
(map car (named-children block))
(length (parameters block))
(define cross-block (TransformerDecoderLayer 8 #:heads 2 #:ffn-width 32
                                             #:norm-first? #t
                                             #:activation 'gelu-tanh))
(map car (named-children cross-block))
]}

@deftogether[(@defproc[(transformer-encoder-layer? [v any/c]) boolean?]
              @defproc[(transformer-decoder-layer? [v any/c]) boolean?])]{
Whether @racket[v] was built by @racket[TransformerEncoderLayer] or by
@racket[TransformerDecoderLayer].
}

@subsection[#:tag "attention-transformer-apply"]{Applying a transformer layer}

An encoder layer applies to one sequence, a decoder layer to the sequence
it is producing, the @deftech{target}, and the memory it reads:

@racketblock[
(encoder-layer src keyword-arg ...)
(decoder-layer tgt memory keyword-arg ...)
]

Each answers a tensor the shape of its first argument. The sequences are
laid out as @racket[MultiheadAttention]'s queries are: @tt{[S, N, d-model]}
for @racket[src], @tt{[T, N, d-model]} for @racket[tgt] and
@tt{[S, N, d-model]} for @racket[memory] by default, the batch first with
@racket[#:batch-first?] @racket[#t], and rank two for one unbatched
sequence. The target and the memory may differ in length but not in batch.

Every keyword is optional and is handed to one attention, under the name
PyTorch's @tt{forward} gives it:

@tabular[#:sep @hspace[2]
         #:style 'boxed
         (list (list @bold{keyword} @bold{PyTorch} @bold{reaches}
                     @bold{shape})
               (list @racket[#:mask] @tt{src_mask}
                     @elem{@racket["self-attn"]'s @racket[#:attn-mask]}
                     @elem{@tt{[S, S]} or @tt{[N·heads, S, S]}})
               (list @racket[#:key-padding-mask] @tt{src_key_padding_mask}
                     @elem{@racket["self-attn"]'s
                           @racket[#:key-padding-mask]}
                     @tt{[N, S]})
               (list @racket[#:causal?] @tt{is_causal}
                     @elem{@racket["self-attn"]'s @racket[#:causal?]} "")
               (list @racket[#:tgt-mask] @tt{tgt_mask}
                     @elem{@racket["self-attn"]'s @racket[#:attn-mask]}
                     @elem{@tt{[T, T]} or @tt{[N·heads, T, T]}})
               (list @racket[#:memory-mask] @tt{memory_mask}
                     @elem{@racket["multihead-attn"]'s @racket[#:attn-mask]}
                     @elem{@tt{[T, S]} or @tt{[N·heads, T, S]}})
               (list @racket[#:tgt-key-padding-mask]
                     @tt{tgt_key_padding_mask}
                     @elem{@racket["self-attn"]'s
                           @racket[#:key-padding-mask]}
                     @tt{[N, T]})
               (list @racket[#:memory-key-padding-mask]
                     @tt{memory_key_padding_mask}
                     @elem{@racket["multihead-attn"]'s
                           @racket[#:key-padding-mask]}
                     @tt{[N, S]})
               (list @racket[#:tgt-causal?] @tt{tgt_is_causal}
                     @elem{@racket["self-attn"]'s @racket[#:causal?]} "")
               (list @racket[#:memory-causal?] @tt{memory_is_causal}
                     @elem{@racket["multihead-attn"]'s @racket[#:causal?]}
                     ""))]

@bold{The masks read as @racket[MultiheadAttention]'s do: a boolean
@racket[#t] hides a key}, the opposite of
@racket[scaled-dot-product-attention]'s sense, and a float mask is added
to the scores. The causal flags build the causal mask, as
@racket[MultiheadAttention]'s @racket[#:causal?] does, where PyTorch's
@tt{is_causal} and @tt{tgt_is_causal} only promise that the mask passed
beside them already is one; @racket[#:tgt-causal?] @racket[#t] is the
usual way to keep a decoder from reading the tokens it predicts. Masks
combine, a key any of them hides staying hidden. An input of another width
than @racket[d-model], a target and memory that disagree on the batch, and
a mask of another shape are contract violations blaming the caller, the
message naming the argument and the shapes.

@torch-examples[
(define src (randn 5 2 8))
(shape (block src))
(define short (eq (tensor '((0 0 0 0 0) (0 0 0 1 1))) 1))
(shape (block src #:key-padding-mask short #:causal? #t))
(define tgt (randn 4 2 8))
(shape (cross-block tgt src #:tgt-causal? #t
                    #:memory-key-padding-mask short))
]

@subsection[#:tag "attention-transformer-pytorch"]{PyTorch's state dicts}

The layers draw their initial values in PyTorch's order: the attentions as
@racket[MultiheadAttention] draws (@secref["attention-multihead-pytorch"]),
@racket["self-attn"] before @racket["multihead-attn"], then
@racket["linear1"] and @racket["linear2"] as @racket[Linear]s draw; the
norms start at one and zero and draw nothing. So under one seed
@racket[(TransformerEncoderLayer e #:heads h)] starts from the values of
@tt{nn.TransformerEncoderLayer(e, h)}, nothing copied across, and so does
the decoder layer.

A checkpoint's keys map onto the parameters by spelling
@tt{self_attn} and @tt{multihead_attn} with a hyphen, @tt{out_proj} as
@tt{out}, and splitting each fused in-projection by rows:

@tabular[#:sep @hspace[2]
         #:style 'boxed
         (list (list @bold{PyTorch} @bold{here})
               (list @tt{self_attn.in_proj_weight}
                     @elem{rows @tt{[0:E]}, @tt{[E:2E]}, @tt{[2E:3E]} as
                           @racket["self-attn.query.weight"],
                           @racket["self-attn.key.weight"],
                           @racket["self-attn.value.weight"]})
               (list @tt{self_attn.in_proj_bias}
                     @elem{thirds, as @racket["self-attn.query.bias"],
                           @racket["self-attn.key.bias"],
                           @racket["self-attn.value.bias"]})
               (list @elem{@tt{self_attn.out_proj.weight}, @tt{.bias}}
                     @elem{@racket["self-attn.out.weight"],
                           @racket["self-attn.out.bias"]})
               (list @tt{multihead_attn.*}
                     @elem{@racket["multihead-attn.*"], split the same way})
               (list @elem{@tt{linear1.*}, @tt{linear2.*}, @tt{norm1.*},
                           @tt{norm2.*}, @tt{norm3.*}}
                     "the same names")
               (list @elem{@tt{layers.N.*}, @tt{norm.*} (stacks)}
                     @elem{@racket["layers.N.*"], @racket["norm.*"], each
                           layer's keys mapped as above}))]

PyTorch's inference fast path, @tt{torch._transformer_encoder_layer_fwd}
and the nested tensors of @tt{TransformerEncoder}, is not mirrored: these
layers always run the computation above. The two agree, with one
exception: given a key-padding mask, PyTorch's nested-tensor stack in eval
mode answers zeros at the padded positions, where the computation above,
and so the library, answers whatever those positions attended to. Nothing
downstream should read them either way. PyTorch's @tt{device} and
@tt{dtype} constructor arguments are @racket[to] here.

@section[#:tag "attention-transformer-stacks"]{Transformer stacks}

@defmodule[torch/nn #:link-target? #f]

@deftogether[(@defproc[(TransformerEncoder
                        [make-layer (and/c (not/c layer?)
                                           (-> transformer-encoder-layer?))]
                        [#:layers layers exact-positive-integer?]
                        [#:norm norm (or/c #f layer?) #f]
                        [#:copies? copies? boolean? #t])
                       transformer-encoder?]
              @defproc[(TransformerDecoder
                        [make-layer (and/c (not/c layer?)
                                           (-> transformer-decoder-layer?))]
                        [#:layers layers exact-positive-integer?]
                        [#:norm norm (or/c #f layer?) #f]
                        [#:copies? copies? boolean? #t])
                       transformer-decoder?])]{
PyTorch's @tt{nn.TransformerEncoder} and @tt{nn.TransformerDecoder}:
@racket[layers] transformer layers applied in turn, then @racket[norm]
when there is one. A stack applies with exactly its layers' arguments and
keywords (@secref["attention-transformer-apply"]), hands every one of them
to every layer, and answers the last layer's output, normalized.

@racket[make-layer] is a procedure of no arguments that builds one layer.
PyTorch's constructor takes a layer and deep-copies it @tt{num_layers}
times, so every layer of a fresh stack starts from the same values. Here
the stack calls @racket[make-layer] once, drawing that layer's values, and
builds the other layers as copies of it, which draw nothing. So under one
seed @racket[(TransformerEncoder make-layer #:layers n)] starts from the
values of @tt{nn.TransformerEncoder(layer, n)} and leaves the random
stream where PyTorch leaves it. Passing a layer itself, as PyTorch's
signature would suggest, is a contract violation: wrap it in a
@racket[lambda].

With @racket[copies?] @racket[#f] the stack calls @racket[make-layer] once
per layer instead, so each draws its own values. That is how GPT-2 and BERT
build their stacks, a list of independently initialized layers, and what
PyTorch's documentation advises doing by hand after constructing a
@tt{TransformerEncoder}.

@racket[norm] is PyTorch's @tt{norm}, usually
@racket[(LayerNorm d-model)]; a stack of pre-norm layers needs it, since
their last sum is never normalized otherwise. The children are
@racket["layers"], a @racket[LayerList], and @racket["norm"] when there is
one, so the parameter names are PyTorch's: @racket["layers.0.linear1.weight"]
and so on (@secref["attention-transformer-pytorch"]).

@torch-examples[
(manual-seed! 0)
(define encoder
  (TransformerEncoder (lambda () (TransformerEncoderLayer 8 #:heads 2
                                                          #:ffn-width 32
                                                          #:norm-first? #t))
                      #:layers 3
                      #:norm (LayerNorm 8)))
(map car (named-children encoder))
(length (parameters encoder))
(define stacked
  (for/list ([l (in-layers (child-ref encoder "layers"))]) l))
(equal? (tensor->list (car (parameters (car stacked))))
        (tensor->list (car (parameters (cadr stacked)))))
(shape (encoder src #:key-padding-mask short))
(define decoder
  (TransformerDecoder (lambda () (TransformerDecoderLayer 8 #:heads 2
                                                          #:ffn-width 32))
                      #:layers 2
                      #:copies? #f))
(shape (decoder tgt (encoder src) #:tgt-causal? #t))
]}

@deftogether[(@defproc[(transformer-encoder? [v any/c]) boolean?]
              @defproc[(transformer-decoder? [v any/c]) boolean?])]{
Whether @racket[v] was built by @racket[TransformerEncoder] or by
@racket[TransformerDecoder].
}

@section[#:tag "attention-positions"]{Positions and causal masks}

@defmodule[torch/nn #:link-target? #f]

Attention does not see order: shuffle the keys and their values together
and every query's answer stays the same. A transformer learns about
positions only from what is added to its inputs, either a fixed encoding
or a learned one.

@defproc[(sinusoidal-positions
          [positions (or/c exact-nonnegative-integer? tensor?)]
          [width (and/c exact-positive-integer? even?)]
          [#:layout layout (or/c 'interleaved 'halves) 'interleaved]
          [#:device device device/c (default-device)])
         tensor?]{
The fixed encoding of @italic{Attention Is All You Need}: for position
@tt{p} and frequency @tt{i} below @tt{width/2}, the pair
@tt{sin(p·ω@subscript{i})} and @tt{cos(p·ω@subscript{i})}, where
@tt{ω@subscript{i} = 10000@superscript{-2i/width}}. Low frequencies tell
far positions apart and high ones near positions, and a shift by a fixed
distance is a rotation of each pair, which attention can learn to read.

@racket[positions] is a length @tt{L}, for positions @tt{0} through
@tt{L - 1} on @racket[device], or a rank-one tensor of positions, any
numeric dtype, which keeps its own device; a decoder that generates one
token at a time asks for the next position this way. The answer is
float32, @tt{[L, width]}, one row per position, ready to add to embeddings
of that width: it broadcasts over a batch-first @tt{[N, L, width]} batch
as it is, and over a sequence-first @tt{[L, N, width]} one once
@racket[unsqueeze] gives it a batch axis at @racket[1].
@racket[#:device] beside a tensor of positions is a contract violation.

@racket[layout] says where the pairs go. @racket['interleaved], the paper's
and PyTorch's tutorial's, puts @tt{sin} in the even columns and @tt{cos} in
the odd ones. @racket['halves] puts every @tt{sin} first and every
@tt{cos} after, the layout tensor2tensor and the DDPM time embedding use;
the speech example's frames and characters take their positions this way
(@secref["ex-asr"]), and the diffusion UNet's
@racket[sinusoidal-embedding] is this layout over its timesteps. The two
layouts hold the same numbers in another order, so a model trained with one
needs that one.

@torch-examples[
(sinusoidal-positions 3 4)
(sinusoidal-positions 3 4 #:layout 'halves)
(sinusoidal-positions (tensor '(7)) 4)
]}

A learned encoding is an @racket[Embedding] with one row per position up
to the longest sequence, indexed by the positions, as GPT-2, BERT and ViT
do; it needs no layer of its own:

@torch-examples[
(define token-table (Embedding 100 8))
(define position-table (Embedding 16 8))
(define ids (tensor '((5 9 2) (7 1 1))))
(shape (+ (token-table ids)
          (position-table (to-dtype (arange 3) 'int64))))
]

@defproc[(causal-mask [size exact-nonnegative-integer?]
                      [#:device device device/c (default-device)]
                      [#:dtype dtype
                               (or/c 'bool 'float32 'float64 'float16
                                     'bfloat16)
                               'bool])
         tensor?]{
The @tt{[size, size]} mask that hides from each position every later one.
As a @racket['bool] tensor it is @racket[#t] above the diagonal:
@racket[#t] hides, @racket[MultiheadAttention]'s sense, which the
transformer layers' @racket[#:mask] and @racket[#:tgt-mask] share. As a
float tensor it is @racket[-inf.0] above the diagonal and zero elsewhere,
PyTorch's @tt{nn.Transformer.generate_square_subsequent_mask}, which adds
to the scores the same way under either sense.

The layers' @racket[#:causal?] flags build this mask themselves, so it is
needed only to combine with another mask or to pass on explicitly. For
@racket[scaled-dot-product-attention], whose boolean mask means attend,
pass the float form, turn the boolean one around with
@racket[(eq (causal-mask n) 0)], or use its own @racket[#:causal?].

@torch-examples[
(causal-mask 3)
(causal-mask 3 #:dtype 'float32)
(define quiet (TransformerEncoderLayer 8 #:heads 2 #:dropout 0.0))
(< (item (max (abs (- (quiet src #:mask (causal-mask 5))
                      (quiet src #:causal? #t)))))
   1e-6)
]}
