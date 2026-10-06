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
it. This chapter computes it by hand on tensors small enough to read,
hands the same work to the library's fused call, and then builds the rest
of a transformer around it: heads and masks, blocks and stacks, and the
positions that tell a model where each token sits.

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
A transformer wraps it in a block with a feed-forward layer and residual
connections, and stacks the blocks.

@section[#:tag "transformers-blocks"]{Blocks, stacks and positions}

A transformer block gives each position two chances to change. Attention
lets it gather from the other positions; then a small feed-forward
network, the same two @racket[Linear] layers at every position, works on
what it gathered. Each of the two adds its answer to its input rather
than replacing it, a @deftech{residual connection}, so the sequence flows
through the block as a stream that every sublayer only adds to, and a
@racket[LayerNorm] keeps that stream in scale.

@racket[TransformerEncoderLayer] is that block. On the toy batch from the
last section, four wide with two heads, and a feed-forward eight wide
inside:

@torch-examples[
(manual-seed! 0)
(define block
  (TransformerEncoderLayer 4 #:heads 2 #:ffn-width 8 #:dropout 0.0
                           #:batch-first? #t))
(map car (named-children block))
(shape (block tokens))
]

The children are the attention, the feed-forward's two @racket[Linear]s
around a @racket[Dropout], a norm for each sublayer and a dropout after
each, under the names PyTorch gives them. With the dropouts off, the block
is attention and then the feed-forward, each added back and normalized:

@torch-examples[
(define (self-attend x)
  ((child-ref block "self-attn") x x x))
(define (feed-forward x)
  ((child-ref block "linear2") (relu ((child-ref block "linear1") x))))
(define after-attention
  ((child-ref block "norm1") (+ tokens (self-attend tokens))))
(define post-norm
  ((child-ref block "norm2")
   (+ after-attention (feed-forward after-attention))))
(< (item (max (abs (- post-norm (block tokens))))) 1e-6)
]

That is @deftech{post-norm}, the original transformer's arrangement and
BERT's: normalize after adding. GPT-2 and the vision transformer use
@deftech{pre-norm} instead: normalize each sublayer's input and add its
answer to the stream untouched, so the stream itself is never normalized
inside the block. Deep stacks train more steadily that way.
@racket[#:norm-first? #t] chooses it:

@torch-examples[
(manual-seed! 0)
(define pre
  (TransformerEncoderLayer 4 #:heads 2 #:ffn-width 8 #:dropout 0.0
                           #:batch-first? #t #:norm-first? #t))
(define (pre-attend x) ((child-ref pre "self-attn") x x x))
(define (pre-feed x)
  ((child-ref pre "linear2") (relu ((child-ref pre "linear1") x))))
(define stream (+ tokens (pre-attend ((child-ref pre "norm1") tokens))))
(define pre-norm (+ stream (pre-feed ((child-ref pre "norm2") stream))))
(< (item (max (abs (- pre-norm (pre tokens))))) 1e-6)
]

The feed-forward's activation is @racket[relu] unless @racket[#:activation]
says otherwise; GPT-2 uses @racket['gelu-tanh]. The block takes the same
masks as its attention, in the same sense, @racket[#t] hiding a key:
@racket[(block tokens #:key-padding-mask padding #:causal? #t)].

A model is several blocks in a row. @racket[TransformerEncoder] builds the
row in one call: it takes the block's arguments, the number of blocks,
and whether to end with a norm, which a stack of pre-norm blocks needs,
since none of them normalizes its output:

@torch-examples[
(manual-seed! 0)
(define encoder
  (TransformerEncoder 4 #:heads 2 #:ffn-width 8 #:dropout 0.0
                      #:batch-first? #t #:norm-first? #t
                      #:layers 2 #:norm? #t))
(map car (named-children encoder))
(length (named-parameters encoder))
(car (map car (named-parameters encoder)))
(shape (encoder tokens #:key-padding-mask padding))
]

The names are the ones a PyTorch checkpoint of the same stack uses, with
@tt{self_attn} spelled @racket["self-attn"] and its fused in-projection
split into @racket["query"], @racket["key"] and @racket["value"]; the
whole mapping is in @secref["attention-transformer-pytorch"]. Like
PyTorch's, the stack starts every block from the same values, copies of
the first, and training moves them apart. A stack of some other block, a
different final norm, or blocks drawn independently, as GPT-2 draws its
own, takes the general form, @racket[GenericTransformerEncoder], which
builds the row from a procedure that makes one block.

Nothing so far knows where a token sits. Attention compares contents, so
shuffling the tokens only shuffles the answers:

@torch-examples[
(define reversed (flip tokens 1))
(< (item (max (abs (- (flip (encoder reversed) 1) (encoder tokens)))))
   1e-5)
]

The order has to be added to the input. The original transformer adds
fixed waves, a sine and a cosine at each of several frequencies, one row
per position; @racket[sinusoidal-positions] computes them, and the rows
broadcast over the batch:

@torch-examples[
(define waves (sinusoidal-positions 3 4))
waves
(define (placed x) (+ x waves))
(item (max (abs (- (flip (encoder (placed reversed)) 1)
                   (encoder (placed tokens))))))
]

With the positions added, a reversed sentence is a different input, not
the same one reordered, and the answers differ by far more than rounding.
GPT-2, BERT and ViT learn their positions instead, from an
@racket[Embedding] with one row per position, indexed like tokens;
@secref["attention-positions"] shows both.

An encoder–decoder puts the pieces together. The encoder reads a source
sequence into memory; the decoder reads the target so far, under a causal
mask so it cannot peek at the tokens it is learning to predict, and
attends to the memory through a cross-attention that hides the source's
padding. A toy translation model over a vocabulary of ten token ids, eight
wide, with a source batch of two sentences of five tokens, the second
padded after three, and targets of four:

@torch-examples[
(manual-seed! 0)
(define embed (Embedding 10 8))
(define source-encoder
  (TransformerEncoder 8 #:heads 2 #:ffn-width 16 #:dropout 0.0
                      #:batch-first? #t #:layers 2))
(define target-decoder
  (TransformerDecoder 8 #:heads 2 #:ffn-width 16 #:dropout 0.0
                      #:batch-first? #t #:layers 2))
(define to-vocabulary (Linear 8 10))
(define source (tensor '((1 4 6 2 3) (5 7 2 0 0))))
(define source-padding (eq source 0))
(define target (tensor '((1 8 9 3) (1 6 6 2))))
(define (embedded ids)
  (+ (embed ids) (sinusoidal-positions (cadr (shape ids)) 8)))
(define (translate source target)
  (define memory
    (source-encoder (embedded source) #:key-padding-mask source-padding))
  (to-vocabulary
   (target-decoder (embedded target) memory
                   #:tgt-causal? #t
                   #:memory-key-padding-mask source-padding)))
(shape (translate source target))
]

The answer is a score for every token of the vocabulary at every target
position, which a cross-entropy loss against the targets shifted by one
would train. The causal mask is what makes that training honest: change
the last target token, and every earlier position's scores stay put.

@torch-examples[
(define changed (tensor '((1 8 9 7) (1 6 6 5))))
(< (item (max (abs (- (narrow (translate source target) 1 0 3)
                      (narrow (translate source changed) 1 0 3)))))
   1e-6)
]

That is a transformer: attention, a feed-forward, residual connections and
norms in a block; blocks in a stack; positions added to the input; and
masks to say who may read whom.

@section[#:tag "transformers-gpt"]{A language model from the pieces}

A GPT has no encoder to read from and so no cross-attention. It is an
encoder stack run under a causal mask, so that every position predicts
the next token from the ones before it. @secref["ex-gpt"] builds a
character-level one from this chapter's pieces and trains it on a
novella. Its whole stack is one call:

@racketblock[
(TransformerEncoder n-embd
                    #:heads n-head
                    #:layers n-layer
                    #:ffn-width (* 4 n-embd)
                    #:activation 'gelu
                    #:norm-first? #t
                    #:dropout 0.0
                    #:batch-first? #t
                    #:norm? #t)
]

That is @tt{nn.TransformerEncoder(nn.TransformerEncoderLayer(...),
n_layer, norm=nn.LayerNorm(n_embd))} with the same settings: pre-norm
blocks as GPT-2 has them, a feed-forward four times the model's width
through @racket[gelu], and the final norm a pre-norm stack needs. Around
it the example puts the rest of this chapter: learned positions, an
@racket[Embedding] indexed by position and added to the token
embeddings; @racket[#:causal? #t] on every application; and a
@racket[Linear] head that turns each position's output into a score for
every character, trained by @racket[cross-entropy] against the text
shifted by one.

The chapter's PyTorch twin builds the same model from
@tt{nn.TransformerEncoder} and matches it under one seed, so the example
doubles as a check that these layers train as PyTorch's do.
@secref["ex-asr"] uses the other arrangement, an encoder and a decoder,
the decoder reading an encoded utterance through cross-attention.
