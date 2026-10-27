#lang scribble/lp2

@(require torch/scribblings/labels)

@title[#:tag "ex-asr"]{Speech to text on LibriSpeech with CTC and attention}

The speech capstone: a hybrid CTC/attention recognizer over LibriSpeech
utterances, the whole arc composed --- FLAC decode
(@racket[librispeech]), the @racket[log-mel-spectrogram] front-end, a
dilated-convolution encoder crowned with bidirectional self-attention, an
autoregressive character decoder attending across into the audio, and
@racket[wer]/@racket[cer] scoring.

The two losses split the work. CTC needs no alignments: the encoder's
per-frame head emits @tt{vocab + blank} distributions and
@racket[ctc-loss] marginalizes over every monotonic alignment between
frames and reference characters, the blank soaking up silence and stretch.
But CTC assumes each frame votes independently --- it cannot spell. The
attention decoder can: it generates characters one at a time, each
conditioned on the characters so far @emph{and} on whatever encoder frames
its cross-attention chooses to look at. Trained together
(@tt{loss = 0.3 ctc + 0.7 ce}, the ESPnet recipe), CTC's monotonic
pressure keeps the encoder honest while attention learns to spell:
CTC aligns, attention spells.

@chunk[<r07-require>
(require (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in racket/string string-split)
         torch torch/nn
         (only-in torch/audio/data audio-info)
         (only-in torch/audio/functional
                  edit-distance log-mel-spectrogram mel-filterbank)
         (only-in torch/audio/librispeech
                  librispeech librispeech-fixture-utterance
                  librispeech-utterances load-utterance utterance-path
                  utterance-transcript)
         (only-in torch/data/loader dataloader)
         (only-in torch/audio/metrics cer wer)
         (only-in torch/text/data decode encode text->vocab))]

@chunk[<r07-provide>
(provide asr asr-loader pick-device run-example greedy-decode transcribe
         mel-frames utterance-features hybrid-batch-loss train-librispeech
         evaluate)]

The transformer half of the model is the library's: the encoder and
decoder blocks are @racket[TransformerEncoderLayer] and
@racket[TransformerDecoderLayer], stacked by @racket[TransformerEncoder]
and @racket[TransformerDecoder], and the attention inside them is
@racket[MultiheadAttention] running @racket[scaled-dot-product-attention]
over every head at once. @secref["guide-transformers"] builds those
layers up from bare tensors and @secref["attention"] documents them; this
chapter is about what a recognizer adds around them, the spectrogram
front end, the masks a batch of unequal utterances needs, and the two
losses.

@bold{Positions without a table.} Attention is permutation-blind, so both
the encoder frames and the decoder characters need a position signal. The
GPT example learned one; here the classic sinusoids come from
@racket[sinusoidal-positions], computed on the fly with no length cap and
no parameters. Frequencies fall geometrically from 1 to 1/10000, and the
@tt{[T, d]} rows broadcast over the batch. @racket[#:layout 'halves] puts
every sine column before every cosine column rather than interleaving
them: the same numbers under a column permutation, in the layout this
model has always used, which the diffusion UNet's time embedding shares.

@bold{Padding masks.} Batching utterances of different lengths means
right-padding them to a rectangle, and neither the convolutions nor the
attention may treat that padding as audio. Two masks do the work, both
built by comparing an @racket[arange] over frame indices against each
row's true length.

@racket[key-padding-mask] marks the padded frames, @tt{[B, T]} and
@racket[#t] where a frame is padding, which is the sense
@racket[MultiheadAttention] and the transformer layers read: a
@racket[#t] key is hidden (@secref["attention-multihead-apply"]). The
encoder's self-attention takes it as @racket[#:key-padding-mask] and the
decoder's cross-attention as @racket[#:memory-key-padding-mask], and the
layers broadcast it over every head and query. Only keys are ever masked:
a padded @emph{query} row still attends the real keys and produces
garbage-but-finite output that the losses ignore, whereas masking whole
rows would leave a row with nothing to attend and breed NaNs. The
decoder's own stream needs no padding mask: with right padding, the
causal mask already stops every real character from seeing pad positions.

@racket[frame-keep] is the convolutional counterpart, a @tt{[B, 1, T]}
multiplier that broadcasts over channels. It exists because an attention
mask applied at the top cannot undo mixing that happened underneath:
every @racket[Conv1d] here carries a bias, so a padded region emerges
from the first convolution holding @emph{bias}-valued activations rather
than zeros, and the next layer --- reaching further with each dilation
--- blends those into the genuine frames near the boundary. The result
would be an utterance whose encoding depends on how long its noisiest
neighbour in the bucket happened to be. Re-zeroing the padding after
every convolution keeps each row's boundary frames identical to what
they would be if the utterance were encoded alone.

@chunk[<r07-mask>
(define (row-lengths lengths)
  (~> (map exact->inexact lengths) tensor (unsqueeze 1)))

(define (key-padding-mask lengths t-len)
  (>= (unsqueeze (arange t-len) 0) (row-lengths lengths)))

(define (frame-keep lengths t-len)
  (reshape (to-dtype (< (unsqueeze (arange t-len) 0)
                        (row-lengths lengths))
                     'float32)
           (length lengths) 1 t-len))]

@bold{The blocks.} An encoder block is the GPT block with the causal mask
deleted: audio is all there at once, so every frame may attend to every
other, forward and backward. It is a @racket[TransformerEncoderLayer]
with the settings GPT-2 made standard, which the model passes its stack
below. @racket[#:norm-first? #t] is pre-norm: each sublayer reads a
normalized view of the residual stream and adds its answer to the stream
untouched. @racket[#:ffn-width] four times the width through
@racket[#:activation 'gelu] is the GPT feed-forward.
@racket[#:dropout 0.0] keeps the encoder free of dropout, as it has always
been here, against the layer's default of 0.1. @racket[#:batch-first? #t]
takes the @tt{[B, T, C]} frames the front end produces. With the child
names the layer gives its pieces, a block under the padding mask computes

@verbatim[#:indent 2]{
x ← x + self-attn(norm1(x), padding)
x ← x + linear2(gelu(linear1(norm2(x))))
}

A decoder block has three sublayers, a @racket[TransformerDecoderLayer]
with the same settings. Causal self-attention comes first: the decoder is
autoregressive over characters, so each position may read only itself
and the characters before it. Then the new move, @emph{cross}-attention,
the layer's @tt{multihead-attn}, where the queries come from the
character stream but the keys and values come from the encoder's
@racket[memory]: each character position reaches across into the audio
and pulls out the frames that sound like it. The only mask there is the
padding mask, hiding the padded audio frames. The feed-forward comes
last, as always:

@verbatim[#:indent 2]{
x ← x + self-attn(norm1(x), causal)
x ← x + multihead-attn(norm2(x), memory, padding)
x ← x + linear2(gelu(linear1(norm3(x))))
}

The decoder's @racket[#:dropout] is the model's, zero by default. Above
zero, the layer drops in PyTorch's places: the attention weights, each
sublayer's answer before it joins the stream, and the feed-forward's
hidden activations. The model drops the decoder stack's output once more
before its head. @secref["transformers-blocks"] checks a pre-norm block's
arithmetic against the layer by hand, and
@secref["attention-transformer-layers"] documents every keyword.

@bold{The model.} The spectrogram side first: two strided @racket[Conv1d]
layers halve time twice (~40ms frames), then four @emph{dilated} residual
convolutions --- dilation 1, 2, 4, 8 --- stretch the receptive field past
a second of context without losing any more time resolution. The frames
transpose to @tt{[B, T', d]}, take their sinusoids, and climb six encoder
blocks. Two heads read the result: the CTC head (@tt{vocab + 1} classes,
blank indexed @emph{after} the characters so ids pass through unshifted)
and the six-block decoder stack. The decoder embeds characters from a
@tt{vocab + 2} table --- @tt{eos} at @racket[vocab-size], @tt{sos} one
past it --- and its head predicts @tt{vocab + 1} classes: characters or
@tt{eos}, never @tt{sos}. The forward takes the audio batch, the
teacher-forced character input, and the list of true frame counts
(@racket[#f] when nothing is padded), from which it derives the
convolution multiplier at each downsampling stage and the padding mask,
and returns both heads' views. The multiplier re-zeroes the padding after
every convolution: each convolution adds a bias, so the padded frames come
out nonzero, and the next kernel would blend them into the real frames at
the boundary.

@racket[TransformerEncoder] builds the encoder stack in one call, as
@tt{nn.TransformerEncoder(nn.TransformerEncoderLayer(...), 6,
norm=nn.LayerNorm(n_embd))} builds PyTorch's: the width, the heads and
the block settings above, six blocks, and @racket[#:norm #t] for the
final @racket[LayerNorm], which a pre-norm stack needs because no block
normalizes the stream it passes on. @racket[TransformerDecoder] builds
the decoder stack the same way, PyTorch's @tt{nn.TransformerDecoder}
over @tt{nn.TransformerDecoderLayer}, with the model's dropout. Like
PyTorch's, each stack starts every block as a copy of its first, and
training moves them apart; for independently drawn blocks, pass the stack
a procedure that builds one block instead of the width
(@secref["attention-transformer-stacks"]). The decoder stack runs with
@racket[#:tgt-causal? #t], which every block hands to its
self-attention, and both stacks take the padding mask.

The parameter paths are PyTorch's (@secref["attention-transformer-pytorch"]):
@tt{dilations.3.weight}, @tt{encoder.layers.0.self-attn.query.weight},
@tt{encoder.norm.weight}, @tt{decoder.layers.5.multihead-attn.out.bias}
and @tt{decoder.layers.0.linear1.weight}. The hand-written blocks this
chapter used before named the same pieces
@tt{encoders.0.attention.wq.weight}, @tt{ln-enc.weight},
@tt{decoders.5.cross.wo.bias} and @tt{decoders.0.mlp.fc1.weight}, with
the same 273 tensors, and drew different initial values: one draw per
block, where @racket[MultiheadAttention] starts its query, key and value
from one xavier draw with zero biases. A checkpoint saved before the
change does not load into this model: @racket[load-state!] refuses it,
naming the missing and unexpected keys. Retrain with
@filepath{scripts/train-asr.rkt}. The keyword defaults are the
fixture-scale configuration the parity twin trains;
@racket[train-librispeech] passes something wider.

@chunk[<r07-model>
(define-layer asr (n-embd conv1 conv2 dilations encoder ctc-head
                   tok-emb decoder hdrop head)
  #:init (n-mels vocab-size
          #:n-embd [n-embd 64]
          #:n-head [n-head 4]
          #:dropout [p-drop 0.0])
  (unless (even? n-embd)
    (error 'asr "n-embd ~a must split into sine/cosine halves" n-embd))
  (set! conv1 (Conv1d n-mels n-embd 3 #:stride 2 #:padding 1))
  (set! conv2 (Conv1d n-embd n-embd 3 #:stride 2 #:padding 1))
  (set! dilations
        (LayerList (for/list ([d '(1 2 4 8)])
                     (Conv1d n-embd n-embd 3 #:dilation d #:padding d))))
  (set! encoder (TransformerEncoder n-embd
                                    #:heads n-head
                                    #:layers 6
                                    #:ffn-width (* 4 n-embd)
                                    #:activation 'gelu
                                    #:norm-first? #t
                                    #:dropout 0.0
                                    #:batch-first? #t
                                    #:norm #t))
  (set! ctc-head (Linear n-embd (add1 vocab-size)))
  (set! tok-emb (Embedding (+ vocab-size 2) n-embd))
  (set! decoder (TransformerDecoder n-embd
                                    #:heads n-head
                                    #:layers 6
                                    #:ffn-width (* 4 n-embd)
                                    #:activation 'gelu
                                    #:norm-first? #t
                                    #:dropout p-drop
                                    #:batch-first? #t
                                    #:norm #t))
  (set! hdrop (Dropout #:p p-drop))
  (set! head (Linear n-embd (add1 vocab-size)))
  #:forward (x dec-in lengths)
  (with-default-device (device x)
    (define (halve n) (quotient (add1 n) 2))
    (define t1 (~> (shape x) caddr halve))
    (define t2 (halve t1))
    (define l1 (and lengths (map halve lengths)))
    (define l2 (and l1 (map halve l1)))
    (define (clip v lens t) (if lens (* v (frame-keep lens t)) v))
    (define c (~> (conv1 x) relu (clip l1 t1)))
    (define c0 (~> (conv2 c) relu (clip l2 t2)))
    (define c4
      (for/fold ([h c0]) ([dil (in-layers dilations)])
        (~> (dil h) relu (+ h) (clip l2 t2))))
    (define padding (and l2 (key-padding-mask l2 t2)))
    (define (positioned v)
      (~> (shape v)
          cadr
          (sinusoidal-positions n-embd #:layout 'halves)
          (+ v)))
    (define memory
      (~> (transpose c4 1 2) positioned (encoder #:key-padding-mask padding)))
    (values (log-softmax (ctc-head memory) 2)
            (~> (tok-emb dec-in)
                positioned
                (decoder memory
                         #:tgt-causal? #t
                         #:memory-key-padding-mask padding)
                hdrop
                head))))]

@bold{The device.} As in the earlier capstones: take the accelerator and
let @racket[with-default-device] scope it, so parameters and batches land
together. Both accelerators run this model natively. The one exception is
@racket[ctc-loss] on dev-clean's longest bucket, whose transcripts pass 511
characters: MPS computes that one loss on the CPU, since its own kernel's
gradients are wrong there. That is a temporary workaround for an upstream
bug (@hyperlink["https://github.com/bkc39/rktorch/issues/258"]{#258}).

@chunk[<r07-device>
(define (pick-device)
  (accelerator-if-available))]

@bold{Features and teacher forcing.} One utterance becomes an
@tt{[80, T]} matrix of mel frames, and a @tt{[1, 80, T]} batch on its
own. The mel filters depend only on the sample rate, so the 16 kHz bank
LibriSpeech needs is built once, as a module-level tensor, and handed to
@racket[log-mel-spectrogram] with @racket[#:filterbank]; any other rate
builds its own. For training, the transcript becomes two shifted
id sequences: the decoder @emph{reads} @tt{[sos, chars]} and must
@emph{predict} @tt{[chars, eos]}. The strided front end downsamples 4x,
so a length @racket[t] signal yields @racket[(downsampled-length t)]
encoder frames --- the per-utterance CTC input lengths.

@chunk[<r07-features>
(define bank-16k
  (mel-filterbank #:n-freqs 201 #:n-mels 80 #:sample-rate 16000 #:device 'cpu))

(define (mel-frames samples rate)
  (log-mel-spectrogram
   (ref samples 0)
   #:sample-rate rate
   #:filterbank (if (= rate 16000)
                    bank-16k
                    (mel-filterbank #:n-freqs 201 #:n-mels 80 #:sample-rate rate
                                    #:device (device samples)))))

(define (utterance-features samples rate)
  (unsqueeze (mel-frames samples rate) 0))

(define (downsampled-length t)
  (~> (add1 t) (quotient 2) add1 (quotient 2)))

(define (transcript-ids vocab transcript)
  (map inexact->exact (tensor->list (encode vocab transcript))))]

@bold{The hybrid loss over a padded batch.} The batch arrives padded:
each utterance's mel matrix padded with zero frames to the widest and
stacked to @tt{[B, 80, T]}, beside the true frame count of every row.
The id sequences pad here --- the decoder input with
@tt{eos} (never attended by anything the loss reads), the decoder target
with @tt{-100} (the @racket[cross-entropy] ignore index, so padded
positions contribute nothing), the CTC targets with @tt{0} (only the
first @tt{target-length} entries of each row are ever read).
@racket[ctc-loss] then gets the @emph{true} per-row frame and character
counts, and the frame counts travel into the forward so the padding is
hidden from the convolutions and from attention alike.
An utterance spoken faster than the encoder's frame rate --- more
characters than downsampled frames --- has @emph{no} valid CTC alignment
and an infinite loss by definition; @racket[#:zero-infinity?] zeroes
those (and their gradients) instead of letting one degenerate utterance
NaN the parameters mid-epoch. The loss is exported, so a caller can reach
it outside whatever extent built the net; it runs on the mels' device.

@chunk[<r07-loss>
(define ctc-weight 0.3)

(define (pad-row ids fill s-max)
  (append ids (make-list (- s-max (length ids)) fill)))

(define (hybrid-batch-loss net vocab x frame-lengths transcripts)
  (when (null? frame-lengths)
    (error 'hybrid-batch-loss "no frame lengths in the batch"))
  (unless (= (car (shape x)) (length frame-lengths) (length transcripts))
    (error 'hybrid-batch-loss "~a rows, ~a frame lengths and ~a transcripts"
           (car (shape x)) (length frame-lengths) (length transcripts)))
  (with-default-device (device x)
    (define v-size (vector-length vocab))
    (define eos v-size)
    (define sos (add1 v-size))
    (define batched? (< 1 (length frame-lengths)))
    (define ids-rows
      (for/list ([tr (in-list transcripts)]) (transcript-ids vocab tr)))
    (define target-lengths (map length ids-rows))
    (define s-max (apply max target-lengths))
    (define (rows->int64 rows)
      (to-dtype (tensor rows) 'int64))
    (define dec-in
      (rows->int64 (for/list ([ids (in-list ids-rows)])
                     (pad-row (cons sos ids) eos (add1 s-max)))))
    (define dec-out
      (rows->int64 (for/list ([ids (in-list ids-rows)])
                     (pad-row (append ids (list eos)) -100 (add1 s-max)))))
    (define ctc-targets
      (rows->int64 (for/list ([ids (in-list ids-rows)])
                     (pad-row ids 0 s-max))))
    (define-values (ctc-lp logits)
      (net x dec-in (and batched? frame-lengths)))
    (define loss-ctc
      (ctc-loss (transpose ctc-lp 0 1) ctc-targets
                #:input-lengths (map downsampled-length frame-lengths)
                #:target-lengths target-lengths
                #:blank v-size
                #:zero-infinity? #t))
    (define loss-ce
      (cross-entropy (reshape logits -1 (add1 v-size))
                     (reshape dec-out -1)))
    (+ (* ctc-weight loss-ctc)
       (* (- 1.0 ctc-weight) loss-ce))))]

@bold{The loader.} @racket[asr-loader] sorts utterances by their frame
counts --- read from FLAC headers via @racket[audio-info], no decode ---
so each @racket[b]-sized batch pads its members to nearly-equal lengths
and the rectangle wastes little. The @racket[librispeech] dataset over
them decodes each utterance when its batch is asked for and featurizes
it with @racket[mel-frames], and its default collate pads and stacks the
mels and keeps their true lengths. The batches come in sorted order,
unshuffled; @racket[w] worker threads build them ahead of the loop.

@chunk[<r07-loader>
(define (utterance-frames u)
  (define-values (frames _rate _channels) (audio-info (utterance-path u)))
  frames)

(define (asr-loader utterances #:batch-size [b 16] #:workers [w 0])
  (define sorted (sort utterances < #:key utterance-frames #:cache-keys? #t))
  (dataloader (librispeech sorted #:transform mel-frames)
              #:batch-size b
              #:workers w))]

@bold{The deterministic core.} @racket[run-example] is the seeded,
offline entry the test harness and the PyTorch parity twin both drive:
5 @racket[adam] steps of the hybrid loss on the committed MISTER QUILTER
fixture --- a batch of one from @racket[asr-loader], so no padding and no
padding mask, moved to the device once --- at the fixture-scale
defaults. The twin builds its stacks as
@tt{nn.TransformerEncoder} and @tt{nn.TransformerDecoder} with the same
settings, and under one seed both sides draw the same initial values,
declaration order being draw order.

@chunk[<r07-run>
(define (run-example #:steps [steps 5] #:device [device (pick-device)])
  (with-default-device device
    (manual-seed! 0)
    (match-define (list x lengths transcripts)
      (for/first ([(x lengths transcripts)
                   (asr-loader (list (librispeech-fixture-utterance))
                               #:batch-size 1)])
        (list (to-device x device) lengths transcripts)))
    (define vocab (text->vocab (car transcripts)))
    (define net (asr 80 (vector-length vocab)))
    (define opt (adam (parameters net) #:lr 0.001))
    (define losses
      (for/list ([_ (in-range steps)])
        (zero-grads! opt)
        (define loss (hybrid-batch-loss net vocab x lengths transcripts))
        (backward! loss)
        (step! opt)
        (item loss)))
    (values losses net vocab device)))]

@bold{Two ways to read the model out.} @racket[greedy-decode] is the CTC
path: argmax the encoder head per frame, collapse consecutive repeats,
drop blanks (collapsing @emph{before} dropping is what lets a blank
separate a genuine double letter). @racket[transcribe] is the attention
path: generate from @tt{sos} one character at a time, feeding each choice
back in, until @tt{eos} or one character per encoder frame --- the
autoregressive loop of the GPT capstone's @racket[generate] with the
audio riding along in cross-attention. The frame count comes from the
input's shape by @racket[downsampled-length], so the cap needs no forward
pass. Both read the device off the model's first parameter, since a
model's tensors are colocated. The script prints both; watching CTC's
phonetic stutter next to attention's spelling is the payoff.

@chunk[<r07-decode>
(define (greedy-decode net vocab features)
  (define v-size (vector-length vocab))
  (define dev (~> (parameters net) car device))
  (define x (to-device features dev))
  (with-default-device dev
    (in-eval-mode net
      (with-no-grad
        (define sos-in
          (~> (tensor (list (add1 v-size))) (to-dtype 'int64) (unsqueeze 0)))
        (define-values (ctc-lp _logits) (net x sos-in #f))
        (define ids
          (map inexact->exact (tensor->list (argmax ctc-lp 2))))
        (define kept
          (for/fold ([prev #f] [acc '()] #:result (reverse acc))
                    ([id (in-list ids)])
            (values id
                    (if (or (equal? id prev) (= id v-size))
                        acc
                        (cons id acc)))))
        (decode vocab kept)))))]

@chunk[<r07-transcribe>
(define (transcribe net vocab features #:max-steps [max-steps #f])
  (define v-size (vector-length vocab))
  (define eos v-size)
  (define sos (add1 v-size))
  (define dev (~> (parameters net) car device))
  (define x (to-device features dev))
  (with-default-device dev
    (in-eval-mode net
      (with-no-grad
        (define cap
          (or max-steps (~> (shape x) caddr downsampled-length)))
        (define (next-id ids)
          (define dec-in
            (~> (tensor ids) (to-dtype 'int64) (unsqueeze 0)))
          (define-values (_ctc-lp logits) (net x dec-in #f))
          (~> (narrow logits 1 (sub1 (length ids)) 1) (argmax 2) item inexact->exact))
        (define ids
          (let loop ([ids (list sos)])
            (define next (next-id ids))
            (cond [(= next eos) (cdr ids)]
                  [(>= (length ids) (add1 cap)) (cdr ids)]
                  [else (loop (append ids (list next)))])))
        (decode vocab ids)))))]

@bold{The real thing.} @racket[train-librispeech] downloads the dev-clean
split (~337MB archive, cached under @envvar{RKTORCH_AUDIO_DIR} or the
system cache dir) and by default trains on @emph{all} of it, in
@racket[batch]-sized batches from @racket[asr-loader]. The spectral front
end stays on the CPU, since an @tt{[80, T]} feature transfer per
utterance is noise next to the model compute, and each padded batch
moves to the training device as it arrives. Nothing is cached between
epochs: every epoch decodes and featurizes the corpus again, which costs
CPU time per epoch but holds no mels in memory, where caching them all
of dev-clean took ~800MB. @racket[#:workers] moves that front end onto
worker threads, ahead of the training loop.
dev-clean is ~5.4 hours of speech --- small for
character-level seq2seq --- so expect recognizable words and partial
spellings, not a production recognizer; the 100-hour train-clean-100
split is the natural next scale.

For calibration, @tt{EPOCHS=40 racket scripts/train-asr.rkt} in the CUDA
shell trains at these defaults on all of dev-clean but the last three
utterances, which the script holds out and scores. On an RTX 3090 Ti the
40 epochs took about ten minutes, fifteen seconds each, when the mels
were cached before the first epoch, and drive the mean hybrid loss from 2.60 in the first epoch to 0.16 in the last. The
CTC head spells phonetically (@tt{STUDFAS} for @emph{steadfast},
@tt{LOWD OF} for @emph{load of}), while the attention decoder emits real
words in roughly the right places (@tt{PRAYES OF MAIN PURAYES} for
@emph{praise of maiden pure}, @tt{WITH THE KARTY SENS} for @emph{with
tardy sense}) but over-generates, and on one utterance falls into a loop
(@tt{THE LOAD OF LOAD OF LOAD OF}) that runs on long past the reference.
Every inserted character counts as an edit, so the three score 1.02 CER
and 1.65 WER, both above one. Three utterances are a small sample, and
one runaway hypothesis dominates them. Scored by @racket[evaluate] on
every 26th utterance of test-clean, a hundred the model never saw, the
same checkpoint's attention decoder lands at 0.79 CER and 1.20 WER, with
9 of the 100 hypotheses running to the step cap; the CTC head's greedy
decode, scored the same way, reaches 0.51 CER.

@chunk[<r07-train>
(define (train-librispeech #:epochs [epochs 20] #:limit [limit #f]
                           #:batch [batch 16]
                           #:n-embd [n-embd 256]
                           #:dropout [p-drop 0.0]
                           #:device [device (pick-device)]
                           #:workers [workers 0]
                           #:log-every [log-every 1])
  (when (and limit (not (exact-positive-integer? limit)))
    (error 'train-librispeech "limit must be a positive integer: ~a" limit))
  (unless (exact-positive-integer? epochs)
    (error 'train-librispeech "epochs must be a positive integer: ~a" epochs))
  (unless (exact-positive-integer? batch)
    (error 'train-librispeech "batch must be a positive integer: ~a" batch))
  (unless (exact-positive-integer? log-every)
    (error 'train-librispeech "log-every must be a positive integer: ~a"
           log-every))
  (unless (exact-nonnegative-integer? workers)
    (error 'train-librispeech "workers must be a non-negative integer: ~a"
           workers))
  (with-default-device device
    (manual-seed! 0)
    (define all (librispeech-utterances "dev-clean"))
    (define utts
      (if (and limit (< limit (length all)))
          (for/list ([u (in-list all)] [_ (in-range limit)]) u)
          all))
    (when (null? utts)
      (error 'train-librispeech "no utterances to train on"))
    (define vocab
      (text->vocab (apply string-append
                          (map utterance-transcript utts))))
    (define net (asr 80 (vector-length vocab) #:n-embd n-embd
                     #:dropout p-drop))
    (define opt (adam (parameters net) #:lr 0.0003))
    (define loader (asr-loader utts #:batch-size batch #:workers workers))
    (for ([epoch (in-range 1 (add1 epochs))])
      (define-values (total steps)
        (for/fold ([total 0.0] [steps 0])
                  ([(x lengths transcripts) loader])
          (zero-grads! opt)
          (define loss
            (hybrid-batch-loss net vocab (to-device x device) lengths
                               transcripts))
          (backward! loss)
          (step! opt)
          (values (+ total (item loss)) (add1 steps))))
      (when (zero? (modulo epoch log-every))
        (printf "epoch ~a/~a: mean loss ~a\n" epoch epochs (/ total steps))
        (flush-output)))
    (values net vocab)))]

@bold{Validation.} Per-utterance rates average badly --- a three-word
reference and a thirty-word one would count equally --- so
@racket[evaluate] accumulates edits and reference lengths across the
whole held-out set and divides once at the end. That is the standard
corpus-level definition of word and character error rate, and it is what
a hyperparameter sweep should compare.

@chunk[<r07-evaluate>
(define (evaluate net vocab utterances)
  (when (null? utterances)
    (error 'evaluate "no utterances to score"))
  (for/fold ([w-edits 0] [w-len 0] [c-edits 0] [c-len 0]
             #:result (values (/ w-edits w-len) (/ c-edits c-len)))
            ([u (in-list utterances)])
    (define-values (samples rate) (load-utterance u))
    (define reference (utterance-transcript u))
    (define hypothesis
      (transcribe net vocab (utterance-features samples rate)))
    (define ref-words (string-split reference))
    (values (+ w-edits (edit-distance ref-words (string-split hypothesis)))
            (+ w-len (length ref-words))
            (+ c-edits (edit-distance (string->list reference)
                                      (string->list hypothesis)))
            (+ c-len (string-length reference)))))]

@bold{Scoring.} Decode an utterance both ways and hold the attention
hypothesis against its reference --- the rates are exact rationals, so a
report like @tt{3/10} reads as literally three word edits over a ten-word
reference:

@racketblock[
(define-values (net vocab) (train-librispeech))
(define-values (samples rate transcript) (load-librispeech-fixture))
(define features (utterance-features samples rate))
(greedy-decode net vocab features)
(define hypothesis (transcribe net vocab features))
(wer transcript hypothesis)
(cer transcript hypothesis)
]

@chunk[<*>
<r07-require>
<r07-provide>
<r07-mask>
<r07-model>
<r07-device>
<r07-features>
<r07-loss>
<r07-loader>
<r07-run>
<r07-decode>
<r07-transcribe>
<r07-train>
<r07-evaluate>]
