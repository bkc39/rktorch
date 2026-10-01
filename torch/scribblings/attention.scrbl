#lang scribble/manual

@(require "common.rkt"
          (for-label (except-in racket/base
                                abs cos exp log sin sort sqrt max min length
                                + - * /)
                     racket/contract
                     torch
                     (only-in torch/nn Dropout)))

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
is @racket[#t]. A float mask is additive in both. @racket[(eq mask 0)]
turns a boolean mask of one sense into the other.

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

Each device chooses its own kernel: on CUDA, FlashAttention for
@racket['float16] and @racket['bfloat16] inputs it supports and the
memory-efficient kernel otherwise, falling back to the computation above.
There is nothing to select here, and the kernels agree up to rounding.
Gradients flow to @racket[query], @racket[key] and @racket[value].

PyTorch's @tt{enable_gqa}, grouped-query attention with fewer key and
value heads than query heads, is not exposed: it stays off.
}
