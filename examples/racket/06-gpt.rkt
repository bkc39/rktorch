#lang scribble/lp2

@(require (for-label (except-in racket/base exp log sqrt max min + - * /)
                     torch torch/nn))

@section[#:tag "ex-gpt"]{Training a char-GPT on Heart of Darkness}

The v3 capstone: a decoder-only transformer language model over characters,
trained on Joseph Conrad's @emph{Heart of Darkness} (Project Gutenberg #219).
The architecture is the standard pre-norm GPT: token + learned positional
embeddings into @racket[n-layer] blocks of (layer-norm @tt{->} causal
self-attention @tt{->} residual) and (layer-norm @tt{->} MLP @tt{->} residual),
then a final layer-norm and a linear head back to vocabulary logits.

Multi-head attention is built @emph{inline} from primitives ---
@racket[Linear] projections, @racket[reshape]/@racket[transpose] head
splitting, @racket[matmul] scores, the @racket[tril]-derived causal mask
through @racket[masked-fill], and @racket[softmax] --- rather than hidden
behind a helper, because watching the tensor shapes move is the point of the
example. (A library-level @tt{TransformerEncoderBlock} is #32.)

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
(provide gpt-block gpt pick-device
         load-excerpt run-example train-excerpt train-novel generate)]

@bold{One transformer block.} Pre-norm, as GPT-2 settled it: the residual
stream is only ever @emph{added to}, each sub-layer reading a normalized view.
Attention first: the normalized input is projected to queries/keys/values,
split into @racket[n-head] heads of @tt{head-dim = n-embd / n-head} (the
@racket[reshape] + @racket[transpose] dance takes @tt{[B, T, C]} to
@tt{[B, H, T, D]}), and scored against itself, scaled by @tt{sqrt(head-dim)}.
The upper triangle of the @tt{[T, T]} score matrix --- pairs where a position
would attend to its own future --- is filled with @tt{-inf} @emph{before}
@racket[softmax], so those weights come out exactly zero: the causal mask that
makes this a language model rather than an oracle. The @tt{[T, T]} bool mask
broadcasts over the batched @tt{[B, H, T, T]} scores. Then the MLP: two
@racket[Linear] layers through @racket[gelu], widened 4x inside, the
GPT-standard shape.

@chunk[<r06-block>
(define-module gpt-block (n-embd n-head)
  #:coerce ([n-head (if (zero? (remainder n-embd n-head))
                        n-head
                        (error 'gpt-block "n-embd ~a not divisible by n-head ~a"
                               n-embd n-head))])
  #:submodules ([ln1 (LayerNorm n-embd)]
                [wq (Linear n-embd n-embd)]
                [wk (Linear n-embd n-embd)]
                [wv (Linear n-embd n-embd)]
                [wo (Linear n-embd n-embd)]
                [ln2 (LayerNorm n-embd)]
                [fc1 (Linear n-embd (* 4 n-embd))]
                [fc2 (Linear (* 4 n-embd) n-embd)])
  #:forward (x)
  (with-default-device (tensor-device x)
    (define shape (tensor-shape x))
    (define batch (car shape))
    (define seq-len (cadr shape))
    (define head-dim (quotient n-embd n-head))
    (define (split-heads m)
      (transpose (reshape m batch seq-len n-head head-dim) 1 2))
    (define xn (ln1 x))
    (define q (split-heads (wq xn)))
    (define k (split-heads (wk xn)))
    (define v (split-heads (wv xn)))
    (define scores (div (matmul q (transpose k 2 3)) (sqrt head-dim)))
    (define causal (eq (tril (ones seq-len seq-len)) 0))
    (define att (softmax (masked-fill scores causal -inf.0) -1))
    (define ctx
      (reshape (transpose (matmul att v) 1 2) batch seq-len n-embd))
    (define x1 (add x (wo ctx)))
    (add x1 (fc2 (gelu (fc1 (ln2 x1)))))))]

@bold{The model.} Token ids gather rows from a learned @racket[Embedding]
table; a second table indexed by @racket[(arange seq-len)] adds a learned
position signal (its @tt{[T, C]} rows broadcast over the batch). The blocks
stack in a @racket[Sequential], whose indexed naming gives PyTorch-style
dotted paths (@tt{blocks.0.ln1.weight}). @racket[block-size] only sizes the
position table --- cropping inputs to fit is the caller's job. Both forwards
scope their temporaries --- the position @racket[arange] here, the
causal-mask @racket[ones] in the block --- to the @emph{input's} device, so
a CUDA-trained net can be applied directly, outside any
@racket[with-default-device] extent, exactly like the Python twin's
@tt{device=idx.device}. The keyword
defaults are the fixture-scale configuration that @racket[run-example] and the
parity twin train; @racket[train-novel] passes something bigger.

@chunk[<r06-model>
(define-module gpt (vocab-size block-size
                    #:n-embd [n-embd 32]
                    #:n-head [n-head 4]
                    #:n-layer [n-layer 2])
  #:submodules ([tok-emb (Embedding vocab-size n-embd)]
                [pos-emb (Embedding block-size n-embd)]
                [blocks (apply Sequential
                               (for/list ([_ (in-range n-layer)])
                                 (gpt-block n-embd n-head)))]
                [ln-f (LayerNorm n-embd)]
                [head (Linear n-embd vocab-size)])
  #:forward (idx)
  (with-default-device (tensor-device idx)
    (define seq-len (cadr (tensor-shape idx)))
    (define pos (to-dtype (arange seq-len) 'int64))
    (~> (add (tok-emb idx) (pos-emb pos))
        blocks ln-f head)))]

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
problem. Full-batch, no shuffling: with a shared seed the @racket[Embedding]
and @racket[Linear] inits draw value-for-value like their @tt{nn.*}
counterparts (declaration order is RNG-draw order on both sides), and the
updates track @tt{torch.optim.Adam} within float tolerance.

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
(64-dim, 2 blocks, 32-char context). On a GPU the default 60 epochs finish in well
under a minute; on CPU it's a few minutes.

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
minibatches --- the batch window just cycles through the text, since a
shuffling loader would need @tt{randperm}, which isn't on the surface yet.
On the CPU this is a coffee-length run; on a GPU it's minutes.

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
  ;; Any parameter's device works (a module's tensors are colocated); the
  ;; context limit comes from pos-emb's row count by *name*, so it survives
  ;; a reordering of gpt's #:submodules list.
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
  <r06-block>
  <r06-model>
  <r06-device>
  <r06-run>
  <r06-train-excerpt>
  <r06-train-novel>
  <r06-generate>]
