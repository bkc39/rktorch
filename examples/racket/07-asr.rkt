#lang scribble/lp2

@(require (for-label (except-in racket/base abs cos exp log sin sqrt max min + - * /)
                     torch torch/nn))

@section[#:tag "ex-asr"]{Speech to text on LibriSpeech with CTC and attention}

The speech capstone: a hybrid CTC/attention recognizer over LibriSpeech
utterances, the whole arc composed --- FLAC decode
(@racket[load-utterance]), the @racket[log-mel-spectrogram] front-end, a
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
         (only-in racket/sequence in-slice)
         (only-in racket/string string-split)
         torch torch/nn
         (only-in torch/audio/data audio-info)
         (only-in torch/audio/functional edit-distance log-mel-spectrogram)
         (only-in torch/audio/librispeech
                  librispeech-utterances load-librispeech-fixture
                  load-utterance utterance-path utterance-transcript)
         (only-in torch/audio/metrics cer wer)
         (only-in torch/data/text decode encode text->vocab))]

@chunk[<r07-provide>
(provide asr asr-encoder-block asr-decoder-block pick-device
         run-example greedy-decode transcribe utterance-features
         hybrid-batch-loss train-librispeech evaluate)]

@bold{Positions without a table.} Attention is permutation-blind, so both
the encoder frames and the decoder characters need a position signal. The
GPT capstone learned one; here the classic sinusoids are computed on the
fly --- no length cap, no parameters, and the @racket[arange], @racket[sin],
@racket[cos], and @racket[exp] primitives this arc ported get their
showcase. Frequencies fall geometrically from 1 to 1/10000; the sine half
and cosine half concatenate to @tt{[T, d]} rows that broadcast over the
batch. (The classic presentation interleaves sine and cosine columns;
concatenating halves is the same code under a column permutation, and both
sides of the parity twin spell it this way.)

@chunk[<r07-positions>
(define (sinusoidal-positions t-len n-embd)
  (define half (quotient n-embd 2))
  (define positions (unsqueeze (arange t-len) 1))
  (define freqs (exp (mul (arange half) (- (/ (log 10000.0) half)))))
  (define angles (mul positions (unsqueeze freqs 0)))
  (cat (list (sin angles) (cos angles)) 1))]

@bold{Padding masks.} Batching utterances of different lengths means
right-padding them to a rectangle, and neither the convolutions nor the
attention may treat that padding as audio. Two masks do the work, both
built by comparing an @racket[arange] over frame indices against each
row's true length.

@racket[key-padding-mask] marks padded @emph{key} columns for attention,
shaped @tt{[B, 1, 1, T]} so it broadcasts over every head and query of
the @tt{[B, H, T, T]} scores. Only keys are ever masked: a padded
@emph{query} row still attends the real keys and produces
garbage-but-finite output that the losses ignore, whereas masking whole
rows would feed @racket[softmax] a row of @tt{-inf} and breed NaNs.
The decoder's own stream needs no such mask: with right padding, the
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
  (unsqueeze (tensor (map exact->inexact lengths)) 1))

(define (key-padding-mask lengths t-len)
  (reshape (ge (unsqueeze (arange t-len) 0) (row-lengths lengths))
           (length lengths) 1 1 t-len))

(define (frame-keep lengths t-len)
  (reshape (to-dtype (lt (unsqueeze (arange t-len) 0)
                         (row-lengths lengths))
                     'float32)
           (length lengths) 1 t-len))]

@bold{The encoder block.} The GPT block with the causal mask deleted:
audio is all there at once, so every frame may attend to every other,
forward and backward. Pre-norm, multi-head attention, then the 4x-widened
@racket[gelu] MLP, residuals throughout. @racket[mask] is @racket[#f] on
the unbatched paths.

@chunk[<r07-encoder-block>
(define-layer asr-encoder-block (n-embd n-head ln1 wq wk wv wo ln2 fc1 fc2)
  #:init (n-embd n-head)
  (unless (and (exact-positive-integer? n-head)
               (zero? (remainder n-embd n-head)))
    (error 'asr-encoder-block
           "n-head ~a must be positive and divide n-embd ~a"
           n-head n-embd))
  (set! ln1 (LayerNorm n-embd))
  (set! wq (Linear n-embd n-embd))
  (set! wk (Linear n-embd n-embd))
  (set! wv (Linear n-embd n-embd))
  (set! wo (Linear n-embd n-embd))
  (set! ln2 (LayerNorm n-embd))
  (set! fc1 (Linear n-embd (* 4 n-embd)))
  (set! fc2 (Linear (* 4 n-embd) n-embd))
  #:forward (x mask)
  (with-default-device (tensor-device x)
    (define batch (car (tensor-shape x)))
    (define seq-len (cadr (tensor-shape x)))
    (define head-dim (quotient n-embd n-head))
    (define (split-heads m)
      (transpose (reshape m batch seq-len n-head head-dim) 1 2))
    (define xn (ln1 x))
    (define q (split-heads (wq xn)))
    (define k (split-heads (wk xn)))
    (define v (split-heads (wv xn)))
    (define scores (div (matmul q (transpose k 2 3)) (sqrt head-dim)))
    (define att
      (softmax (if mask (masked-fill scores mask -inf.0) scores) -1))
    (define ctx
      (reshape (transpose (matmul att v) 1 2) batch seq-len n-embd))
    (define x1 (add x (wo ctx)))
    (add x1 (fc2 (gelu (fc1 (ln2 x1)))))))]

@bold{The decoder block.} Three sub-layers now. Causal self-attention
first --- the decoder is autoregressive over characters, so the
@racket[tril] mask from the GPT block returns. Then the new move:
@emph{cross}-attention, where the queries come from the character stream
but the keys and values come from the encoder's @racket[memory] --- each
character position reaches across into the audio and pulls out the frames
that sound like it. The only mask there is @racket[mem-mask], hiding the
padded audio frames. MLP last, as always.

@chunk[<r07-decoder-block>
(define-layer asr-decoder-block (n-embd n-head p-drop
                                 cdrop ln1 sq sk sv so
                                 ln2 cq ck cv co
                                 ln3 fc1 fc2)
  #:init (n-embd n-head #:dropout [p-drop 0.0])
  (unless (and (exact-positive-integer? n-head)
               (zero? (remainder n-embd n-head)))
    (error 'asr-decoder-block
           "n-head ~a must be positive and divide n-embd ~a"
           n-head n-embd))
  (set! cdrop (Dropout #:p p-drop))
  (set! ln1 (LayerNorm n-embd))
  (set! sq (Linear n-embd n-embd))
  (set! sk (Linear n-embd n-embd))
  (set! sv (Linear n-embd n-embd))
  (set! so (Linear n-embd n-embd))
  (set! ln2 (LayerNorm n-embd))
  (set! cq (Linear n-embd n-embd))
  (set! ck (Linear n-embd n-embd))
  (set! cv (Linear n-embd n-embd))
  (set! co (Linear n-embd n-embd))
  (set! ln3 (LayerNorm n-embd))
  (set! fc1 (Linear n-embd (* 4 n-embd)))
  (set! fc2 (Linear (* 4 n-embd) n-embd))
  #:forward (x memory mem-mask)
  (with-default-device (tensor-device x)
    (define batch (car (tensor-shape x)))
    (define seq-len (cadr (tensor-shape x)))
    (define mem-len (cadr (tensor-shape memory)))
    (define head-dim (quotient n-embd n-head))
    (define (split-heads m len)
      (transpose (reshape m batch len n-head head-dim) 1 2))
    (define xn (ln1 x))
    (define q (split-heads (sq xn) seq-len))
    (define k (split-heads (sk xn) seq-len))
    (define v (split-heads (sv xn) seq-len))
    (define scores (div (matmul q (transpose k 2 3)) (sqrt head-dim)))
    (define causal (eq (tril (ones seq-len seq-len)) 0))
    (define att (softmax (masked-fill scores causal -inf.0) -1))
    (define x1
      (add x (so (reshape (transpose (matmul att v) 1 2)
                          batch seq-len n-embd))))
    ;; skipped outright at p=0 so no RNG is drawn and the twin stays
    ;; value-for-value
    (define x1n (if (zero? p-drop) (ln2 x1) (cdrop (ln2 x1))))
    (define q2 (split-heads (cq x1n) seq-len))
    (define k2 (split-heads (ck memory) mem-len))
    (define v2 (split-heads (cv memory) mem-len))
    (define scores2
      (div (matmul q2 (transpose k2 2 3)) (sqrt head-dim)))
    (define att2
      (softmax (if mem-mask (masked-fill scores2 mem-mask -inf.0) scores2)
               -1))
    (define x2
      (add x1 (co (reshape (transpose (matmul att2 v2) 1 2)
                           batch seq-len n-embd))))
    (add x2 (fc2 (gelu (fc1 (ln3 x2)))))))]

@bold{The model.} The spectrogram side first: two strided @racket[Conv1d]
layers halve time twice (~40ms frames), then four @emph{dilated} residual
convolutions --- dilation 1, 2, 4, 8 --- stretch the receptive field past
a second of context without losing any more time resolution. The frames
transpose to @tt{[B, T', d]}, take their sinusoids, and climb six encoder
blocks (every sub-layer residual). Two heads read the result: the CTC
head (@tt{vocab + 1} classes, blank indexed @emph{after} the characters
so ids pass through unshifted) and the six-block decoder stack. The
decoder embeds characters from a @tt{vocab + 2} table ---
@tt{eos} at @racket[vocab-size], @tt{sos} one past it --- and its head
predicts @tt{vocab + 1} classes: characters or @tt{eos}, never @tt{sos}.
The forward takes the audio batch, the teacher-forced character input,
and the list of true frame counts (@racket[#f] when nothing is padded),
from which it derives the convolution multiplier at each downsampling
stage and the attention key mask, and returns both heads' views. The keyword defaults are the fixture-scale
configuration the parity twin trains; @racket[train-librispeech] passes
something wider.

@chunk[<r07-model>
(define-layer asr (n-embd p-drop
                   conv1 conv2 dil1 dil2 dil3 dil4
                   enc1 enc2 enc3 enc4 enc5 enc6 ln-enc ctc-head
                   tok-emb dec1 dec2 dec3 dec4 dec5 dec6 ln-dec hdrop head)
  #:init (n-mels vocab-size
          #:n-embd [n-embd 64]
          #:n-head [n-head 4]
          #:dropout [p-drop 0.0])
  (unless (even? n-embd)
    (error 'asr "n-embd ~a must split into sine/cosine halves" n-embd))
  (set! conv1 (Conv1d n-mels n-embd 3 #:stride 2 #:padding 1))
  (set! conv2 (Conv1d n-embd n-embd 3 #:stride 2 #:padding 1))
  (set! dil1 (Conv1d n-embd n-embd 3 #:dilation 1 #:padding 1))
  (set! dil2 (Conv1d n-embd n-embd 3 #:dilation 2 #:padding 2))
  (set! dil3 (Conv1d n-embd n-embd 3 #:dilation 4 #:padding 4))
  (set! dil4 (Conv1d n-embd n-embd 3 #:dilation 8 #:padding 8))
  (set! enc1 (asr-encoder-block n-embd n-head))
  (set! enc2 (asr-encoder-block n-embd n-head))
  (set! enc3 (asr-encoder-block n-embd n-head))
  (set! enc4 (asr-encoder-block n-embd n-head))
  (set! enc5 (asr-encoder-block n-embd n-head))
  (set! enc6 (asr-encoder-block n-embd n-head))
  (set! ln-enc (LayerNorm n-embd))
  (set! ctc-head (Linear n-embd (add1 vocab-size)))
  (set! tok-emb (Embedding (+ vocab-size 2) n-embd))
  (set! dec1 (asr-decoder-block n-embd n-head #:dropout p-drop))
  (set! dec2 (asr-decoder-block n-embd n-head #:dropout p-drop))
  (set! dec3 (asr-decoder-block n-embd n-head #:dropout p-drop))
  (set! dec4 (asr-decoder-block n-embd n-head #:dropout p-drop))
  (set! dec5 (asr-decoder-block n-embd n-head #:dropout p-drop))
  (set! dec6 (asr-decoder-block n-embd n-head #:dropout p-drop))
  (set! ln-dec (LayerNorm n-embd))
  (set! hdrop (Dropout #:p p-drop))
  (set! head (Linear n-embd (add1 vocab-size)))
  #:forward (x dec-in lengths)
  (with-default-device (tensor-device x)
    (define (halve n) (quotient (add1 n) 2))
    (define t0 (caddr (tensor-shape x)))
    (define t1 (halve t0))
    (define t2 (halve t1))
    (define l1 (and lengths (map halve lengths)))
    (define l2 (and l1 (map halve l1)))
    ;; re-zero the padding after every convolution: each carries a bias,
    ;; so pad regions come out nonzero and the next kernel would blend
    ;; them into real boundary frames
    (define (clip v lens t) (if lens (mul v (frame-keep lens t)) v))
    (define c (clip (relu (conv1 x)) l1 t1))
    (define c0 (clip (relu (conv2 c)) l2 t2))
    (define c1 (clip (add c0 (relu (dil1 c0))) l2 t2))
    (define c2 (clip (add c1 (relu (dil2 c1))) l2 t2))
    (define c3 (clip (add c2 (relu (dil3 c2))) l2 t2))
    (define c4 (clip (add c3 (relu (dil4 c3))) l2 t2))
    (define enc-mask (and l2 (key-padding-mask l2 t2)))
    (define t-len (caddr (tensor-shape c4)))
    (define e0 (add (transpose c4 1 2) (sinusoidal-positions t-len n-embd)))
    (define memory
      (ln-enc
       (enc6 (enc5 (enc4 (enc3 (enc2 (enc1 e0 enc-mask) enc-mask)
                                enc-mask)
                          enc-mask)
                   enc-mask)
             enc-mask)))
    (define ctc-log-probs (log-softmax (ctc-head memory) 2))
    (define s-len (cadr (tensor-shape dec-in)))
    (define d0 (add (tok-emb dec-in) (sinusoidal-positions s-len n-embd)))
    (define d
      (ln-dec
       (dec6 (dec5 (dec4 (dec3 (dec2 (dec1 d0 memory enc-mask)
                                     memory enc-mask)
                               memory enc-mask)
                         memory enc-mask)
                   memory enc-mask)
             memory enc-mask)))
    (values ctc-log-probs
            (head (if (zero? p-drop) d (hdrop d))))))]

@bold{The device.} As in the earlier capstones: take the accelerator and
let @racket[with-default-device] scope it, so parameters and batches land
together. Both accelerators run this model natively --- libtorch 2.9
registers no MPS @tt{ctc_loss} kernel, but @racket[ctc-loss] marginalizes
that one op on the CPU and hands the gradient back, so Apple silicon
trains on the GPU like CUDA does.

@chunk[<r07-device>
(define (pick-device)
  (accelerator-if-available))]

@bold{Features and teacher forcing.} One utterance becomes a
@tt{[1, 80, T]} batch. For training, the transcript becomes two shifted
id sequences: the decoder @emph{reads} @tt{[sos, chars]} and must
@emph{predict} @tt{[chars, eos]}. The strided front end downsamples 4x,
so a length @racket[t] signal yields @racket[(downsampled-length t)]
encoder frames --- the per-utterance CTC input lengths.

@chunk[<r07-features>
(define (utterance-features samples rate)
  (unsqueeze (log-mel-spectrogram (ref samples 0) #:sample-rate rate) 0))

(define (downsampled-length t)
  (quotient (add1 (quotient (add1 t) 2)) 2))

(define (transcript-ids vocab transcript)
  (map inexact->exact (tensor->list (encode vocab transcript))))]

@bold{The hybrid loss over a padded batch.} Each utterance's mel matrix
pads with zero frames to the widest in the batch and the batch stacks to
@tt{[B, 80, T]}; the id sequences pad likewise --- the decoder input with
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
NaN the parameters mid-epoch.

@chunk[<r07-loss>
(define ctc-weight 0.3)

(define (pad-row ids fill s-max)
  (append ids (make-list (- s-max (length ids)) fill)))

(define (hybrid-batch-loss net vocab mels transcripts)
  (when (null? mels)
    (error 'hybrid-batch-loss "no mels in the batch"))
  (unless (= (length mels) (length transcripts))
    (error 'hybrid-batch-loss "~a mels but ~a transcripts"
           (length mels) (length transcripts)))
  ;; the mels carry the device: this is exported, so callers reach it
  ;; from outside whatever extent built the net
  (with-default-device (tensor-device (car mels))
    (define v-size (vector-length vocab))
    (define eos v-size)
    (define sos (add1 v-size))
    (define frame-lengths
      (for/list ([m (in-list mels)]) (cadr (tensor-shape m))))
    (define t-max (apply max frame-lengths))
    (define x
      (stack (for/list ([m (in-list mels)]
                        [t (in-list frame-lengths)])
               (if (= t t-max)
                   m
                   (cat (list m (zeros (car (tensor-shape m)) (- t-max t)))
                        1)))
             0))
    (define batched? (< 1 (length mels)))
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
    (add (mul ctc-weight loss-ctc)
         (mul (- 1.0 ctc-weight) loss-ce))))]

@bold{The deterministic core.} @racket[run-example] is the seeded,
offline entry the test harness and the PyTorch parity twin both drive:
5 @racket[adam] steps of the hybrid loss on the committed MISTER QUILTER
fixture --- a batch of one, so no padding and no mask --- at the
fixture-scale defaults.

@chunk[<r07-run>
(define (run-example #:steps [steps 5] #:device [device (pick-device)])
  (with-default-device device
    (manual-seed! 0)
    (define-values (samples rate transcript) (load-librispeech-fixture))
    (define vocab (text->vocab transcript))
    (define mel (to-device (ref (utterance-features samples rate) 0) device))
    (define net (asr 80 (vector-length vocab)))
    (define opt (adam (parameters net) #:lr 0.001))
    (define losses
      (for/list ([_ (in-range steps)])
        (zero-grads! opt)
        (define loss (hybrid-batch-loss net vocab (list mel)
                                        (list transcript)))
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
audio riding along in cross-attention. The script prints both; watching
CTC's phonetic stutter next to attention's spelling is the payoff.

@chunk[<r07-decode>
(define (greedy-decode net vocab features)
  (define v-size (vector-length vocab))
  ;; any parameter's device works: a module's tensors are colocated
  (define dev (tensor-device (car (parameters net))))
  (define x (to-device features dev))
  (with-default-device dev
    (in-eval-mode net
      (with-no-grad
        (define sos-in
          (unsqueeze (to-dtype (tensor (list (add1 v-size))) 'int64) 0))
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
  (define dev (tensor-device (car (parameters net))))
  (define x (to-device features dev))
  (with-default-device dev
    (in-eval-mode net
      (with-no-grad
        ;; the cap is the encoder's own frame count, arithmetic on the
        ;; input shape — no forward pass needed to learn it
        (define cap
          (or max-steps (downsampled-length (caddr (tensor-shape x)))))
        (define (next-id ids)
          (define dec-in
            (unsqueeze (to-dtype (tensor ids) 'int64) 0))
          (define-values (_ctc-lp logits) (net x dec-in #f))
          (inexact->exact
           (item (argmax (narrow logits 1 (sub1 (length ids)) 1) 2))))
        (define ids
          (let loop ([ids (list sos)])
            (define next (next-id ids))
            (cond [(= next eos) (cdr ids)]
                  [(>= (length ids) (add1 cap)) (cdr ids)]
                  [else (loop (append ids (list next)))])))
        (decode vocab ids)))))]

@bold{The real thing.} @racket[train-librispeech] downloads the dev-clean
split (~337MB archive, cached under @envvar{RKTORCH_AUDIO_DIR} or the
system cache dir) and by default trains on @emph{all} of it. Utterances
sort by their frame counts --- read from FLAC headers via
@racket[audio-info], no decode --- so each @racket[batch]-sized bucket
pads its members to nearly-equal lengths and the rectangle wastes little.
The spectral front end stays on the CPU --- libtorch-bin 2.9's cuFFT
raises @tt{CUFFT_INTERNAL_ERROR} on this driver stack, and an
@tt{[80, T]} feature transfer per utterance is noise next to the model
compute anyway --- and the mels move to the training device.
dev-clean is ~5.4 hours of speech --- small for
character-level seq2seq --- so expect recognizable words and partial
spellings, not a production recognizer; the 100-hour train-clean-100
split is the natural next scale.

For calibration, 40 epochs at these defaults on an RTX 3090 Ti take
about twenty minutes and drive the hybrid loss from 2.57 to 0.11. On
held-out utterances that lands around @tt{0.6} CER --- the CTC head
spelling phonetically (@tt{ARKTHRIS} for @emph{Arcturus},
@tt{STEUDFAS} for @emph{steadfast}) while the attention decoder emits
real words in roughly the right places. Word error rate stays just
above 1.0, because the decoder over-generates and every insertion
counts against it.

@chunk[<r07-train>
(define (train-librispeech #:epochs [epochs 20] #:limit [limit #f]
                           #:batch [batch 16]
                           #:n-embd [n-embd 256]
                           #:dropout [p-drop 0.0]
                           #:device [device (pick-device)]
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
  (with-default-device device
    (manual-seed! 0)
    (define all (librispeech-utterances "dev-clean"))
    (define utts
      (if (and limit (< limit (length all)))
          (for/list ([u (in-list all)] [_ (in-range limit)]) u)
          all))
    (define sorted
      (sort utts <
            #:key (lambda (u)
                    (define-values (frames _rate _channels)
                      (audio-info (utterance-path u)))
                    frames)
            #:cache-keys? #t))
    (define buckets
      (for/list ([b (in-slice batch (in-list sorted))]) b))
    (when (null? buckets)
      (error 'train-librispeech "no utterances to train on"))
    (define vocab
      (text->vocab (apply string-append
                          (map utterance-transcript utts))))
    ;; constructed before the featurization pass so a bad width fails
    ;; immediately rather than after ~800MB of preprocessing
    (define net (asr 80 (vector-length vocab) #:n-embd n-embd
                     #:dropout p-drop))
    (define opt (adam (parameters net) #:lr 0.0003))
    ;; decode + featurize once, cache the mels on the training device
    ;; (~800MB for all of dev-clean) so every epoch is pure model compute
    (define bucket-data
      (for/list ([bucket (in-list buckets)])
        (cons (for/list ([u (in-list bucket)])
                (define-values (samples rate) (load-utterance u))
                (to-device (ref (utterance-features samples rate) 0)
                           device))
              (map utterance-transcript bucket))))

    (for ([epoch (in-range 1 (add1 epochs))])
      (define-values (total steps)
        (for/fold ([total 0.0] [steps 0])
                  ([bd (in-list bucket-data)])
          (zero-grads! opt)
          (define loss
            (hybrid-batch-loss net vocab (car bd) (cdr bd)))
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
<r07-positions>
<r07-mask>
<r07-encoder-block>
<r07-decoder-block>
<r07-model>
<r07-device>
<r07-features>
<r07-loss>
<r07-run>
<r07-decode>
<r07-transcribe>
<r07-train>
<r07-evaluate>]
