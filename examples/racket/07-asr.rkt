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
(require torch torch/nn
         (only-in torch/audio/functional log-mel-spectrogram)
         (only-in torch/audio/librispeech
                  librispeech-utterances load-librispeech-fixture
                  load-utterance utterance-transcript)
         (only-in torch/audio/metrics cer wer)
         (only-in torch/data/text decode encode text->vocab))]

@chunk[<r07-provide>
(provide asr asr-encoder-block asr-decoder-block pick-device
         run-example greedy-decode transcribe utterance-features
         train-librispeech)]

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

@bold{The encoder block.} The GPT block with the causal mask deleted:
audio is all there at once, so every frame may attend to every other,
forward and backward. Pre-norm, 4-head attention, then the 4x-widened
@racket[gelu] MLP, residuals throughout.

@chunk[<r07-encoder-block>
(define-module asr-encoder-block (n-embd n-head)
  #:coerce ([n-head (if (zero? (remainder n-embd n-head))
                        n-head
                        (error 'asr-encoder-block
                               "n-embd ~a not divisible by n-head ~a"
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
    (define att (softmax scores -1))
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
that sound like it. No mask there: any character may listen to any frame.
MLP last, as always.

@chunk[<r07-decoder-block>
(define-module asr-decoder-block (n-embd n-head)
  #:coerce ([n-head (if (zero? (remainder n-embd n-head))
                        n-head
                        (error 'asr-decoder-block
                               "n-embd ~a not divisible by n-head ~a"
                               n-embd n-head))])
  #:submodules ([ln1 (LayerNorm n-embd)]
                [sq (Linear n-embd n-embd)]
                [sk (Linear n-embd n-embd)]
                [sv (Linear n-embd n-embd)]
                [so (Linear n-embd n-embd)]
                [ln2 (LayerNorm n-embd)]
                [cq (Linear n-embd n-embd)]
                [ck (Linear n-embd n-embd)]
                [cv (Linear n-embd n-embd)]
                [co (Linear n-embd n-embd)]
                [ln3 (LayerNorm n-embd)]
                [fc1 (Linear n-embd (* 4 n-embd))]
                [fc2 (Linear (* 4 n-embd) n-embd)])
  #:forward (x memory)
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
    (define x1n (ln2 x1))
    (define q2 (split-heads (cq x1n) seq-len))
    (define k2 (split-heads (ck memory) mem-len))
    (define v2 (split-heads (cv memory) mem-len))
    (define scores2
      (div (matmul q2 (transpose k2 2 3)) (sqrt head-dim)))
    (define att2 (softmax scores2 -1))
    (define x2
      (add x1 (co (reshape (transpose (matmul att2 v2) 1 2)
                           batch seq-len n-embd))))
    (add x2 (fc2 (gelu (fc1 (ln3 x2)))))))]

@bold{The model.} The spectrogram side first: two strided @racket[Conv1d]
layers halve time twice (~40ms frames), then three @emph{dilated} residual
convolutions --- dilation 1, 2, 4 --- stretch the receptive field toward
half a second of context without losing any more time resolution. The
frames transpose to @tt{[B, T', d]}, take their sinusoids, and pass
through two encoder blocks. Two heads read the result: the CTC head
(@tt{vocab + 1} classes, blank indexed @emph{after} the characters so ids
pass through unshifted) and the decoder stack. The decoder embeds
characters from a @tt{vocab + 2} table --- @tt{eos} at @racket[vocab-size],
@tt{sos} one past it --- and its head predicts @tt{vocab + 1} classes:
characters or @tt{eos}, never @tt{sos}. The forward takes the audio batch
and the teacher-forced character input and returns both heads' views.

@chunk[<r07-model>
(define-module asr (n-mels vocab-size
                    #:n-embd [n-embd 64]
                    #:n-head [n-head 4])
  #:submodules ([conv1 (Conv1d n-mels n-embd 3 #:stride 2 #:padding 1)]
                [conv2 (Conv1d n-embd n-embd 3 #:stride 2 #:padding 1)]
                [dil1 (Conv1d n-embd n-embd 3 #:dilation 1 #:padding 1)]
                [dil2 (Conv1d n-embd n-embd 3 #:dilation 2 #:padding 2)]
                [dil3 (Conv1d n-embd n-embd 3 #:dilation 4 #:padding 4)]
                [enc1 (asr-encoder-block n-embd n-head)]
                [enc2 (asr-encoder-block n-embd n-head)]
                [ln-enc (LayerNorm n-embd)]
                [ctc-head (Linear n-embd (add1 vocab-size))]
                [tok-emb (Embedding (+ vocab-size 2) n-embd)]
                [dec1 (asr-decoder-block n-embd n-head)]
                [dec2 (asr-decoder-block n-embd n-head)]
                [ln-dec (LayerNorm n-embd)]
                [head (Linear n-embd (add1 vocab-size))])
  #:forward (x dec-in)
  (with-default-device (tensor-device x)
    (define c (relu (conv2 (relu (conv1 x)))))
    (define c1 (add c (relu (dil1 c))))
    (define c2 (add c1 (relu (dil2 c1))))
    (define c3 (add c2 (relu (dil3 c2))))
    (define t-len (caddr (tensor-shape c3)))
    (define memory
      (ln-enc
       (enc2 (enc1 (add (transpose c3 1 2)
                        (sinusoidal-positions t-len n-embd))))))
    (define ctc-log-probs (log-softmax (ctc-head memory) 2))
    (define s-len (cadr (tensor-shape dec-in)))
    (define d
      (ln-dec
       (dec2 (dec1 (add (tok-emb dec-in)
                        (sinusoidal-positions s-len n-embd))
                   memory)
             memory)))
    (values ctc-log-probs (head d))))]

@bold{The device.} As in the earlier capstones.

@chunk[<r07-device>
(define (pick-device)
  (accelerator-if-available))]

@bold{Features and teacher forcing.} One utterance becomes a
@tt{[1, 80, T]} batch. For training, the transcript becomes two shifted
id sequences: the decoder @emph{reads} @tt{[sos, chars]} and must
@emph{predict} @tt{[chars, eos]}.

@chunk[<r07-features>
(define (utterance-features samples rate)
  (unsqueeze (log-mel-spectrogram (ref samples 0) #:sample-rate rate) 0))

(define (transcript-ids vocab transcript)
  (map inexact->exact (tensor->list (encode vocab transcript))))

(define (teacher-pair vocab transcript)
  (define v-size (vector-length vocab))
  (define ids (transcript-ids vocab transcript))
  (values
   (unsqueeze (to-dtype (tensor (cons (add1 v-size) ids)) 'int64) 0)
   (unsqueeze (to-dtype (tensor (append ids (list v-size))) 'int64) 0)))]

@bold{The hybrid loss.} Both heads against the same transcript:
@racket[ctc-loss] over the encoder frames (time-major, input length read
off the model's own output shape) and teacher-forced @racket[cross-entropy]
over the decoder's next-character logits, mixed 0.3/0.7.

@chunk[<r07-loss>
(define ctc-weight 0.3)

(define (hybrid-loss net vocab x transcript)
  (define v-size (vector-length vocab))
  (define-values (dec-in dec-out) (teacher-pair vocab transcript))
  (define targets (unsqueeze (encode vocab transcript) 0))
  (define-values (ctc-lp logits) (net x dec-in))
  (define loss-ctc
    (ctc-loss (transpose ctc-lp 0 1) targets
              #:input-lengths (list (cadr (tensor-shape ctc-lp)))
              #:target-lengths (list (string-length transcript))
              #:blank v-size))
  (define loss-ce
    (cross-entropy (reshape logits -1 (add1 v-size))
                   (reshape dec-out -1)))
  (add (mul ctc-weight loss-ctc)
       (mul (- 1.0 ctc-weight) loss-ce)))]

@bold{The deterministic core.} @racket[run-example] is the seeded,
offline entry the test harness and the PyTorch parity twin both drive:
5 @racket[adam] steps of the hybrid loss on the committed MISTER QUILTER
fixture, at the fixture-scale defaults.

@chunk[<r07-run>
(define (run-example #:steps [steps 5] #:device [device (pick-device)])
  (with-default-device device
    (manual-seed! 0)
    (define-values (samples rate transcript) (load-librispeech-fixture))
    (define vocab (text->vocab transcript))
    (define x (to-device (utterance-features samples rate) device))
    (define net (asr 80 (vector-length vocab)))
    (define opt (adam (parameters net) #:lr 0.001))
    (define losses
      (for/list ([_ (in-range steps)])
        (zero-grads! opt)
        (define loss (hybrid-loss net vocab x transcript))
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
        (define-values (ctc-lp _logits) (net x sos-in))
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
        (define (step ids)
          (define dec-in
            (unsqueeze (to-dtype (tensor ids) 'int64) 0))
          (define-values (ctc-lp logits) (net x dec-in))
          (values (cadr (tensor-shape ctc-lp))
                  (inexact->exact
                   (item (argmax (narrow logits 1
                                         (sub1 (length ids)) 1))))))
        (define-values (frame-cap _first) (step (list sos)))
        (define cap (or max-steps frame-cap))
        (define ids
          (let loop ([ids (list sos)])
            (define-values (_frames next) (step ids))
            (cond [(= next eos) (cdr ids)]
                  [(>= (length ids) (add1 cap)) (cdr ids)]
                  [else (loop (append ids (list next)))])))
        (decode vocab ids)))))]

@bold{The real thing.} @racket[train-librispeech] downloads the dev-clean
split (~337MB archive, cached under @envvar{RKTORCH_AUDIO_DIR} or the
system cache dir) and by default trains on @emph{all} of it, one
vocabulary over the transcripts, utterance-at-a-time epochs. Batch size
stays 1: utterances vary from 2 to 30+ seconds, and padding them into
rectangles buys throughput at the cost of masking bookkeeping that
belongs to #87's dataloader, not this example. dev-clean is ~5.4 hours
of speech --- small for character-level seq2seq --- so expect recognizable
words and partial spellings, not a production recognizer; the 100-hour
train-clean-100 split is the natural next scale.

@chunk[<r07-train>
(define (train-librispeech #:epochs [epochs 20] #:limit [limit #f]
                           #:n-embd [n-embd 128]
                           #:device [device (pick-device)]
                           #:log-every [log-every 1])
  (with-default-device device
    (manual-seed! 0)
    (define all (librispeech-utterances "dev-clean"))
    (define utts
      (if (and limit (< limit (length all)))
          (for/list ([u (in-list all)] [_ (in-range limit)]) u)
          all))
    (define vocab
      (text->vocab (apply string-append
                          (map utterance-transcript utts))))
    (define net (asr 80 (vector-length vocab) #:n-embd n-embd))
    (define opt (adam (parameters net) #:lr 0.0003))
    (for ([epoch (in-range 1 (add1 epochs))])
      (define-values (total steps)
        (for/fold ([total 0.0] [steps 0])
                  ([u (in-list utts)])
          (define-values (samples rate) (load-utterance u))
          (zero-grads! opt)
          (define loss
            (hybrid-loss net vocab
                         (to-device (utterance-features samples rate)
                                    device)
                         (utterance-transcript u)))
          (backward! loss)
          (step! opt)
          (values (+ total (item loss)) (add1 steps))))
      (when (zero? (modulo epoch log-every))
        (printf "epoch ~a/~a: mean loss ~a\n" epoch epochs (/ total steps))))
    (values net vocab)))]

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
<r07-encoder-block>
<r07-decoder-block>
<r07-model>
<r07-device>
<r07-features>
<r07-loss>
<r07-run>
<r07-decode>
<r07-transcribe>
<r07-train>]
