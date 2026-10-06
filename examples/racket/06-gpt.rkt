#lang scribble/lp2

@(require (for-label (except-in racket/base abs cos exp log sin sort sqrt max min length + - * /)
                     torch torch/nn))

@title[#:tag "ex-gpt"]{Training a char-GPT on Heart of Darkness}

A decoder-only transformer language model over characters, trained on
Joseph Conrad's @emph{Heart of Darkness} (Project Gutenberg #219). The
architecture is the standard pre-norm GPT: token and learned position
embeddings, then @racket[n-layer] blocks that each run causal
self-attention and a feed-forward network on the residual stream, then a
final layer norm and a linear head back to vocabulary logits.

Every piece of that is a library layer. Each block is a
@racket[TransformerEncoderLayer], the blocks are stacked by
@racket[TransformerEncoder], and the attention inside them is
@racket[MultiheadAttention] running @racket[scaled-dot-product-attention]
over every head at once. @secref["guide-transformers"] builds those layers
up from bare tensors; this chapter puts them to work, so its code is the
model's arrangement and its training loop rather than the arithmetic of
attention.

@chunk[<r06-require>
(require racket/runtime-path
         (only-in racket/file file->string)
         (only-in racket/list take-right)
         torch torch/nn
         (only-in torch/data/text
                  contiguous-blocks
                  decode
                  encode
                  load-heart-of-darkness
                  load-text-fixture
                  text->vocab))]

@chunk[<r06-provide>
(provide gpt pick-device load-excerpt run-example train-excerpt train-novel
         generate)]

@bold{The blocks.} A GPT block is what PyTorch calls a transformer
@emph{encoder} layer: self-attention and then a feed-forward network, with
nothing to read from an encoder. (That third sublayer, cross-attention
over an encoder's output, is what a @racket[TransformerDecoderLayer] adds,
and a decoder-only model has no encoder.) Two things make the layer a GPT
block: the causal mask the model applies it under, and four settings the
model passes its stack below. @racket[#:norm-first? #t] is pre-norm, as
GPT-2 settled it: each sublayer reads a normalized view of the residual
stream and adds its answer to the stream untouched. @racket[#:ffn-width]
four times the width and @racket[#:activation 'gelu] make the
feed-forward the GPT-standard MLP, widened 4x inside through the exact
@racket[gelu] (GPT-2 trained with its tanh approximation,
@racket['gelu-tanh]; a model trained from scratch has no reason to prefer
either). @racket[#:dropout 0.0] turns off the dropout PyTorch's default
applies in four places, so the seeded parity twin compares arithmetic
rather than random masks. And @racket[#:batch-first? #t] takes batches as
the @tt{[B, T, C]} the embeddings produce.

Written out, with the child names a @racket[TransformerEncoderLayer] gives
its pieces, a block applied under the causal mask computes

@verbatim[#:indent 2]{
x ← x + self-attn(norm1(x), causal)
x ← x + linear2(gelu(linear1(norm2(x))))
}

where @tt{self-attn} projects its input through its @tt{query}, @tt{key}
and @tt{value} children, splits each projection into @racket[n-head] heads
of @tt{n-embd / n-head}, attends every head at once with later positions
hidden, joins the heads and projects them back through @tt{out}.
@secref["transformers-blocks"] checks a pre-norm block's arithmetic
against the layer by hand, and @secref["attention-transformer-layers"]
documents every keyword.

@bold{The model.} Token ids gather rows from a learned @racket[Embedding]
table; a second table indexed by @racket[(arange seq-len)] adds a learned
position signal, its @tt{[T, C]} rows broadcasting over the batch
(@secref["attention-positions"]). @racket[TransformerEncoder] builds the
whole stack in one call, as
@tt{nn.TransformerEncoder(nn.TransformerEncoderLayer(...), n_layer,
norm=nn.LayerNorm(n_embd))} builds PyTorch's: the width, the heads and
the block settings above, @racket[n-layer] blocks, and @racket[#:norm? #t]
for the final @racket[LayerNorm], which a pre-norm stack needs because no
block normalizes the stream it passes on. Like PyTorch's, the stack starts
every block as a copy of the first, and training moves them apart. (GPT-2
draws each block afresh instead; @racket[GenericTransformerEncoder] with
@racket[#:copies? #f] builds that stack,
@secref["attention-transformer-stacks"].) The forward applies the stack
with @racket[#:causal? #t], which every block hands to its attention: each
position may attend only to itself and the positions before it, the mask
that makes this a language model rather than an oracle.

The parameter paths are PyTorch's (@secref["attention-transformer-pytorch"]):
@tt{transformer.layers.0.norm1.weight} is the first block's attention
layer norm, @tt{transformer.layers.0.self-attn.query.weight} its query
projection, @tt{transformer.layers.0.linear1.weight} its MLP's first layer,
and @tt{transformer.norm.weight} the final norm. Those paths are new: the
hand-written blocks this chapter used before named the same pieces
@tt{blocks.0.attention.norm.weight}, @tt{blocks.0.attention.branch.wq.weight}
and @tt{blocks.0.mlp.branch.fc1.weight}, and drew different initial values
(@racket[MultiheadAttention] starts its query, key and value from one
xavier draw with zero biases). A checkpoint saved before the change does
not load into this model: @racket[load-state!] refuses it, naming the
missing and unexpected keys. Retrain with @filepath{scripts/train-gpt.rkt}.

@racket[block-size] only sizes the position table --- cropping inputs to
fit is the caller's job. The forward scopes the position
@racket[arange] to the @emph{input's} device, so a CUDA-trained net can be
applied directly, outside any @racket[with-default-device] extent, exactly
like the Python twin's @tt{device=idx.device}; the causal flag is not a
tensor and needs no placing. The keyword defaults are the fixture-scale
configuration that @racket[run-example] and the parity twin train;
@racket[train-novel] passes something bigger.

@chunk[<r06-model>
(define-layer gpt (tok-emb pos-emb transformer head)
  #:init (vocab-size block-size
          #:n-embd [n-embd 32]
          #:n-head [n-head 4]
          #:n-layer [n-layer 2])
  (set! tok-emb (Embedding vocab-size n-embd))
  (set! pos-emb (Embedding block-size n-embd))
  (set! transformer (TransformerEncoder n-embd
                                        #:heads n-head
                                        #:layers n-layer
                                        #:ffn-width (* 4 n-embd)
                                        #:activation 'gelu
                                        #:norm-first? #t
                                        #:dropout 0.0
                                        #:batch-first? #t
                                        #:norm? #t))
  (set! head (Linear n-embd vocab-size))
  #:forward (idx)
  (with-default-device (tensor-device idx)
    (define seq-len (cadr (tensor-shape idx)))
    (define pos (to-dtype (arange seq-len) 'int64))
    (~> (+ (tok-emb idx) (pos-emb pos))
        (transformer #:causal? #t)
        head)))]

@bold{The device.} As in the MNIST capstone: pick the accelerator when one is
present, and let @racket[with-default-device] scope it so parameters and
batches land together.

@chunk[<r06-device>
(define (pick-device)
  (accelerator-if-available))]

@bold{The deterministic core.} @racket[run-example] is the seeded, offline
entry the test harness and the PyTorch parity twin both drive: the committed
841-char fixture becomes @racket[contiguous-blocks] of 16 chars, and a
fixture-scale @racket[gpt] trains for @racket[steps] full-batch @racket[adam]
steps. The next-char loss is @racket[cross-entropy] with the @tt{[B, T, V]}
logits and @tt{[B, T]} targets flattened to one @tt{[B*T]}-row classification
problem. Full-batch, no shuffling: with a shared seed the
@racket[Embedding], @racket[TransformerEncoder] and @racket[Linear]
inits draw value-for-value like their @tt{nn.*} counterparts (declaration
order is RNG-draw order on both sides, and the twin builds its stack as
@tt{nn.TransformerEncoder(nn.TransformerEncoderLayer(...), 2,
norm=nn.LayerNorm(32))} with the same settings), and the updates track
@tt{torch.optim.Adam} within float tolerance.

@chunk[<r06-run>
(define fixture-block-size 16)

(define (run-example #:steps [steps 5] #:device [device (pick-device)])
  (with-default-device device
    (manual-seed! 0)
    (define text (load-text-fixture))
    (define vocab (text->vocab text))
    (define-values (xs ys)
      (contiguous-blocks (encode vocab text) fixture-block-size))
    (define net (gpt (vector-length vocab) fixture-block-size))
    (define opt (adam (parameters net) #:lr 0.001))
    (define losses
      (for/list ([_ (in-range steps)])
        (zero-grads! opt)
        (define logits (net xs))
        (define loss (cross-entropy (reshape logits -1 (vector-length vocab))
                                    (reshape ys -1)))
        (backward! loss)
        (step! opt)
        (item loss)))
    (values losses net vocab device)))]

@bold{The middle path: offline training on a committed excerpt.} Between the
841-char parity fixture (too small to learn from) and the full-novella
download sits @filepath{examples/data/heart-of-darkness-part-i.txt}: the
opening ~31k characters of Part I, committed to the repo, so this trains a
real --- if small --- language model with @emph{no network at all}. The loop
is epoch-shaped: sequential batch-stride passes over the excerpt's
contiguous blocks, with the ragged trailing remainder --- fewer than
@racket[batch] rows; 4 of 964 here --- dropped each epoch, the same
tail-drop semantics as @racket[train-novel] and the train script
(re-training the final @racket[n - batch] window instead would overlap
most of it with the previous window every epoch, a worse bias than
skipping under half a percent of the data). The per-epoch mean loss prints
so the run is watchable; the model is scaled down to match the data
(64-dim, 2 blocks, 32-char context). The default 60 epochs take about
twenty seconds on an RTX 3090 Ti and about as long on an eight-core CPU:
the model is too small to keep a GPU busy. The mean loss falls from
about 2.3 at the tenth epoch to about 0.8 at the sixtieth.

@chunk[<r06-train-excerpt>
(define-runtime-path excerpt-path "../data/heart-of-darkness-part-i.txt")

(define (load-excerpt)
  (file->string excerpt-path))

(define (train-excerpt #:epochs [epochs 60] #:batch [batch 32]
                       #:block-size [block-size 32]
                       #:device [device (pick-device)]
                       #:log-every [log-every 10])
  (with-default-device device
    (manual-seed! 0)
    (define text (load-excerpt))
    (define vocab (text->vocab text))
    (define v-size (vector-length vocab))
    (define-values (xs ys)
      (contiguous-blocks (encode vocab text) block-size))
    (define n (car (tensor-shape xs)))
    (unless (<= batch n)
      (error 'train-excerpt "batch ~a exceeds the excerpt's ~a blocks"
             batch n))
    (define net (gpt v-size block-size #:n-embd 64 #:n-head 4 #:n-layer 2))
    (define opt (adam (parameters net) #:lr 0.001))
    (for ([epoch (in-range 1 (add1 epochs))])
      (define-values (total steps)
        (for/fold ([total 0.0] [steps 0])
                  ([start (in-range 0 (add1 (- n batch)) batch)])
          (zero-grads! opt)
          (define loss
            (cross-entropy
             (reshape (net (narrow xs 0 start batch)) -1 v-size)
             (reshape (narrow ys 0 start batch) -1)))
          (backward! loss)
          (step! opt)
          (values (+ total (item loss)) (add1 steps))))
      (when (zero? (modulo epoch log-every))
        (printf "epoch ~a/~a: mean loss ~a\n" epoch epochs (/ total steps))))
    (values net vocab)))]

@bold{The real thing.} @racket[train-novel] downloads the full novella
(cached under @envvar{RKTORCH_TEXT_DIR} or the system cache dir; the Project
Gutenberg boilerplate is stripped by the loader), carves it into ~3300
64-char blocks, and trains a 4-layer model on deterministic contiguous
minibatches --- the batch window cycles through the text in order, the
same sweep as the training script's epochs. The default 2000 steps take
about 40 seconds on an RTX 3090 Ti and about four minutes on an
eight-core CPU, the loss falling from 4.4 or 4.5 at the first step to
about 1.4.

@chunk[<r06-train-novel>
(define (train-novel #:steps [steps 2000] #:batch [batch 64]
                     #:block-size [block-size 64]
                     #:device [device (pick-device)]
                     #:log-every [log-every 100])
  (with-default-device device
    (manual-seed! 0)
    (define text (load-heart-of-darkness))
    (define vocab (text->vocab text))
    (define-values (xs ys) (contiguous-blocks (encode vocab text) block-size))
    (define n (car (tensor-shape xs)))
    (unless (<= batch n)
      (error 'train-novel "batch ~a exceeds the corpus's ~a blocks" batch n))
    (define net (gpt (vector-length vocab) block-size
                     #:n-embd 128 #:n-head 4 #:n-layer 4))
    (define opt (adam (parameters net) #:lr 0.0003))
    ;; Sequential wraparound sweep, aligned with scripts/train-gpt.rkt's
    ;; epoch loop: batch-stride windows tile the corpus and every one is
    ;; visited each `windows` steps. (A (* step batch)-mod-M cycle only
    ;; covers all offsets when gcd(batch, M) = 1 — at the novella's size it
    ;; would skip most of them, including the final window.) The trailing
    ;; partial window (< batch blocks) is dropped, as in the script.
    (define windows (quotient n batch))
    (for ([step (in-range steps)])
      (define start (* batch (modulo step windows)))
      (zero-grads! opt)
      (define loss
        (cross-entropy
         (reshape (net (narrow xs 0 start batch)) -1 (vector-length vocab))
         (reshape (narrow ys 0 start batch) -1)))
      (backward! loss)
      (step! opt)
      (when (zero? (modulo step log-every))
        (printf "step ~a: loss ~a\n" step (item loss))))
    (values net vocab)))]

@bold{Generation.} Autoregressive and greedy: run the context through the
model, @racket[argmax] the logits at the @emph{last} position, append, repeat
--- cropping the context to the trailing ids the position table can address.
Both defaults are @emph{derived from the net itself} rather than hardcoded:
the device from where its parameters live (@racket[with-default-device] only
steers @emph{newly created} tensors, so the rollout context must be built
where the weights already are --- a @racket[train-novel] net on an accelerator
would
otherwise device-mismatch), and the context limit from the position table's
row count, looked up as @tt{"pos-emb.weight"} in
@racket[named-parameters] (a 64-block net would otherwise be silently
cropped to the fixture's 16). Greedy sampling is deterministic (no
temperature knob to seed), which is what the smoke test wants; it also
produces the characteristically repetitive prose greedy decoding is known
for, which is half the fun. Inference-only, so the model runs under
@racket[in-eval-mode] and @racket[with-no-grad] --- no autograd graph, and
the prior training mode is restored on the way out. The prompt must be
non-empty (checked here --- there is no position to read logits from
otherwise) and drawn from the training vocabulary (@racket[encode] errors
on any character outside it).

@chunk[<r06-generate>
(define (generate net vocab prompt
                  #:steps [steps 256]
                  #:block-size [block-size #f]
                  #:device [device #f])
  (when (zero? (string-length prompt))
    (error 'generate "prompt must be non-empty"))
  ;; Any parameter's device works (a model's tensors are colocated); the
  ;; context limit comes from pos-emb's row count by *name*, so it survives
  ;; a reordering of gpt's #:init body.
  (define dev (or device (tensor-device (car (parameters net)))))
  (define ctx-limit
    (or block-size
        (car (tensor-shape
              (cdr (assoc "pos-emb.weight" (named-parameters net)))))))
  (with-default-device dev
    (in-eval-mode net
      (with-no-grad
        (define start
          (map inexact->exact (tensor->list (encode vocab prompt))))
        (define ids
          (for/fold ([ids start]) ([_ (in-range steps)])
            (define ctx (take-right ids (min (length ids) ctx-limit)))
            (define idx
              (reshape (to-dtype (tensor ctx) 'int64) 1 (length ctx)))
            (define logits (net idx))
            (define next-logits (narrow logits 1 (- (length ctx) 1) 1))
            (define next (inexact->exact (item (argmax next-logits))))
            (append ids (list next))))
        (decode vocab ids)))))]

@chunk[<*>
  <r06-require>
  <r06-provide>
  <r06-model>
  <r06-device>
  <r06-run>
  <r06-train-excerpt>
  <r06-train-novel>
  <r06-generate>]
