#lang scribble/lp2

@(require (for-label (except-in racket/base abs cos exp log sin sort sqrt max min length + - * /)
                     torch torch/nn torch/data/loader torch/data/translation))

@section[#:tag "ex-translation"]{Translating French with a GRU and attention}

The sequence-to-sequence model of the PyTorch tutorial, which is also
ocaml-torch's @tt{translation} example: a @racket[GRU] encoder reads a French
sentence into a sequence of states, and a @racket[GRU] decoder writes the
English one word at a time, looking back over the encoder's states through
Bahdanau attention at every step. The data is the tutorial's too, the short
sentence pairs @racketmodname[torch/data/translation] prepares.

Everything recurrent in the library meets here: the layers' carried state (the
decoder is one @racket[GRU] applied a step at a time), @racket[topk] for greedy
decoding, @racket[nll-loss] with an ignored padding index,
@racket[clip-grad-norm!], and a loop whose shape depends on a coin flipped per
batch.

@chunk[<r13-require>
(require (only-in racket/list last [take list-take])
         (only-in racket/string string-split)
         torch torch/nn
         torch/data/loader
         torch/data/translation
         (only-in torch/audio/functional edit-distance))]

@chunk[<r13-provide>
(provide encoder attention decoder seq2seq
         pick-device run-example train-pairs train-translator
         translate token-error-rate split-pairs)]

@bold{The encoder.} An @racket[Embedding] and a batch-first @racket[GRU]. It
answers every step's output, @tt{[B, L, H]}, which the decoder will attend
over, and the state that starts the decoder, @tt{[1, B, H]}. That state is not
the @racket[GRU]'s final one: a batch is padded to a common width, so for a
short sentence the final state has also read the padding after it. The state
wanted is the one at the sentence's @tt{<eos>}, and for a one-layer GRU that is
simply the output at that position. A one-hot mask of the @tt{<eos>} positions,
@tt{[B, 1, L]}, picks that row out of the @tt{[B, L, H]} outputs with one
batched @racket[matmul].

@chunk[<r13-encoder>
(define-layer encoder (embed drop gru)
  #:init (vocab-size hidden dropout)
  (set! embed (Embedding vocab-size hidden))
  (set! drop (Dropout #:p dropout))
  (set! gru (GRU hidden hidden #:batch-first? #t))
  #:forward (tokens)
  (define-values (outputs _final) (gru (drop (embed tokens))))
  (define at-eos
    (unsqueeze (to-dtype (eq tokens eos-id) (tensor-dtype outputs)) 1))
  (values outputs (transpose (matmul at-eos outputs) 0 1)))]

@bold{Attention.} Bahdanau's additive score: project the decoder's state (the
query) and the encoder's outputs (the keys) into a common space, add them,
squash with @racket[tanh], and reduce each position to one number. The query
is @tt{[B, 1, H]} and the keys @tt{[B, L, H]}, so the sum broadcasts over the
@racket[L] source positions. Positions that hold padding get a score of
@racket[-inf.0], so @racket[softmax] gives them no weight at all. The context
is the weighted sum of the keys, which a batched @racket[matmul] of the
@tt{[B, 1, L]} weights with the @tt{[B, L, H]} keys computes in one call.

@chunk[<r13-attention>
(define-layer attention (wa ua va)
  #:init (hidden)
  (set! wa (Linear hidden hidden))
  (set! ua (Linear hidden hidden))
  (set! va (Linear hidden 1))
  #:forward (query keys padding)
  (define scores (transpose (va (tanh (+ (wa query) (ua keys)))) 1 2))
  (define weights (softmax (masked-fill scores padding -inf.0) -1))
  (values (matmul weights keys) weights))]

@bold{One decoder step.} Embed the previous word, ask the attention for a
context using the current state as the query (the state is @tt{[1, B, H]} and
the query wants the batch first, hence the @racket[transpose]), and feed the
two side by side to the @racket[GRU] as a sequence of length one, together with
the state. The new output goes through a @racket[Linear] head to vocabulary
logits. The step answers the logits, the new state and the attention weights.

@chunk[<r13-decoder>
(define-layer decoder (embed drop attend gru head)
  #:init (vocab-size hidden dropout)
  (set! embed (Embedding vocab-size hidden))
  (set! drop (Dropout #:p dropout))
  (set! attend (attention hidden))
  (set! gru (GRU (* 2 hidden) hidden #:batch-first? #t))
  (set! head (Linear hidden vocab-size))
  #:forward (previous state keys padding)
  (define embedded (drop (embed previous)))
  (define-values (context weights)
    (attend (transpose state 0 1) keys padding))
  (define-values (output next-state)
    (gru (cat (list embedded context) 2) state))
  (values (head output) next-state weights))]

@bold{The whole model.} Encode, then unroll the decoder for as many steps as
the target is wide, starting from @racket[sos-id]. What each step receives as
its previous word is the one decision in the loop. With a @racket[targets]
tensor and @racket[teacher-forcing?] set it is the true previous word, which
keeps early training on the rails; otherwise it is the decoder's own best
guess, @racket[topk] with @racket[k] = 1 over the logits, detached so no
gradient flows through the choice. That second mode is also exactly greedy
decoding, so translation below is this same forward with no targets. The
step logits are joined into @tt{[B, L, V]}.

@chunk[<r13-model>
(define-layer seq2seq (enc dec)
  #:init (source-size target-size
          #:hidden [hidden 128]
          #:dropout [dropout 0.1])
  (set! enc (encoder source-size hidden dropout))
  (set! dec (decoder target-size hidden dropout))
  #:forward (sources targets steps teacher-forcing?)
  (define-values (keys encoded) (enc sources))
  (define padding (unsqueeze (eq sources pad-id) 1))
  (define batch (car (tensor-shape sources)))
  (define start
    (full-like (narrow sources 1 0 1) sos-id))
  (define-values (logits _previous _state)
    (for/fold ([logits '()] [previous start] [state encoded])
              ([step (in-range steps)])
      (define-values (step-logits next-state _weights)
        (dec previous state keys padding))
      (define next
        (if teacher-forcing?
            (narrow targets 1 step 1)
            (let-values ([(_top best) (topk step-logits 1)])
              (detach (reshape best batch 1)))))
      (values (cons step-logits logits) next next-state)))
  (cat (reverse logits) 1))]

@bold{The loss.} The mean negative log likelihood of the target words, over
real positions only: @racket[nll-loss] skips every target equal to
@racket[pad-id], so a short sentence in a padded batch is not rewarded for
predicting padding.

@chunk[<r13-loss>
(define (translation-loss logits targets)
  (define vocab-size (last (tensor-shape logits)))
  (nll-loss (log-softmax (reshape logits -1 vocab-size) 1)
            (reshape targets -1)
            #:ignore-index pad-id))]

@bold{One step of training.} The coin for teacher forcing is one uniform draw
from the CPU's seeded stream per batch, compared with @racket[forcing]; at
0.5, half the batches are decoded from the truth and half from the model's own
output. The gradient is clipped before the update.

@chunk[<r13-step>
(define (train-step! net opt sources targets #:forcing [forcing 0.5])
  (define teacher-forcing? (< (item (rand 1 #:device 'cpu)) forcing))
  (zero-grads! opt)
  (define steps (cadr (tensor-shape targets)))
  (define loss
    (translation-loss (net sources targets steps teacher-forcing?) targets))
  (backward! loss)
  (clip-grad-norm! (parameters net) 1.0)
  (step! opt)
  loss)]

@chunk[<r13-device>
(define (pick-device)
  (accelerator-if-available))]

@bold{The deterministic core.} @racket[run-example] trains on the 287
committed pairs, full batch, for five steps from seed 0, which is what the
test harness and the PyTorch twin drive. Both sides draw their initial weights
in declaration order and their teacher-forcing coins from the same stream, so
the losses and the final parameters agree within float tolerance.

@chunk[<r13-run>
(define sentence-width 10)

(define (run-example #:steps [steps 5] #:device [device (pick-device)])
  (with-default-device device
    (manual-seed! 0)
    (define pairs (load-translation-fixture))
    (define-values (source-vocab target-vocab) (pairs->vocabs pairs))
    (define-values (sources targets)
      (pairs->tensors pairs source-vocab target-vocab #:width sentence-width))
    (define net (seq2seq (vocab-size source-vocab) (vocab-size target-vocab)
                         #:hidden 32 #:dropout 0.0))
    (define opt (adam (parameters net) #:lr 0.001))
    (define xs (to-device sources device))
    (define ys (to-device targets device))
    (define losses
      (for/list ([_ (in-range steps)])
        (item (train-step! net opt xs ys))))
    (values losses net source-vocab target-vocab)))]

@bold{Training for real.} @racket[train-pairs] trains on any list of pairs
with shuffled minibatches from a seeded @racket[dataloader];
@racket[train-translator] holds out a tenth of the tutorial's 11,445 pairs,
trains on the rest and answers the held-out pairs along with the model, so the
caller can measure on sentences the model never saw. The vocabularies come
from all the pairs, held-out ones included, as in the tutorial: a held-out
sentence may combine words in a new way but never contains an unknown word.
The runner, @filepath{examples/test/13-translation.rkt}, does exactly this
(the first run downloads and caches the 3 MB archive), then prints the held-out
token error rate and ten held-out translations; @envvar{EPOCHS} overrides the
epoch count.

@chunk[<r13-train>
(define (split-pairs pairs #:held-out [fraction 0.1] #:seed [seed 0])
  (define order
    (map inexact->exact
         (tensor->list (randperm (length pairs)
                                 #:generator (make-generator seed)))))
  (define shuffled
    (let ([by-index (list->vector pairs)])
      (for/list ([i (in-list order)]) (vector-ref by-index i))))
  (define held (inexact->exact (floor (* fraction (length pairs)))))
  (values (list-tail shuffled held) (list-take shuffled held)))

(define (train-pairs pairs source-vocab target-vocab
                     #:epochs [epochs 30] #:batch [batch 64]
                     #:hidden [hidden 128] #:lr [lr 0.001]
                     #:device [device (pick-device)]
                     #:log-every [log-every 5])
  (with-default-device device
    (manual-seed! 0)
    (define-values (sources targets)
      (pairs->tensors pairs source-vocab target-vocab #:width sentence-width))
    (define loader
      (dataloader (tensor-dataset (to-device sources device)
                                  (to-device targets device))
                  #:batch-size batch #:shuffle? #t
                  #:generator (make-generator 0)))
    (define net (seq2seq (vocab-size source-vocab) (vocab-size target-vocab)
                         #:hidden hidden))
    (define opt (adam (parameters net) #:lr lr))
    (for ([epoch (in-range 1 (add1 epochs))])
      (define-values (total batches)
        (for/fold ([total 0.0] [batches 0])
                  ([(xs ys) (in-dataloader loader)])
          (values (+ total (item (train-step! net opt xs ys)))
                  (add1 batches))))
      (when (zero? (modulo epoch log-every))
        (printf "epoch ~a/~a: mean loss ~a\n" epoch epochs (/ total batches))))
    net))

(define (train-translator #:epochs [epochs 30]
                          #:device [device (pick-device)]
                          #:log-every [log-every 5])
  (define pairs (load-translation-pairs))
  (define-values (source-vocab target-vocab) (pairs->vocabs pairs))
  (define-values (training held-out) (split-pairs pairs))
  (define net
    (train-pairs training source-vocab target-vocab
                 #:epochs epochs #:device device #:log-every log-every))
  (values net source-vocab target-vocab held-out))]

@bold{Translating.} Normalize and encode the sentences, run the model with no
targets so that every step feeds on its own @racket[topk] choice, and take the
@racket[argmax] of each step's logits. @racket[decode-tokens] cuts each row at
its first @tt{<eos>}. It runs under @racket[in-eval-mode], which turns the
dropout off, and @racket[with-no-grad], on the device the weights are on.

@chunk[<r13-translate>
(define (translate net source-vocab target-vocab sentences)
  (define device (tensor-device (car (parameters net))))
  (in-eval-mode net
    (with-no-grad
      (define sources
        (to-device (sentences->tensor source-vocab
                                      (map normalize-sentence sentences)
                                      #:width sentence-width)
                   device))
      (define best
        (to-device (argmax (net sources #f sentence-width #f) 2) 'cpu))
      (for/list ([row (in-range (length sentences))])
        (decode-tokens target-vocab (narrow best 0 row 1))))))]

@bold{Measuring.} BLEU is more machinery than this example wants; the token
error rate is enough to see the model work. It is the word-level
@racket[edit-distance] between each translation and its reference, summed and
divided by the references' total length: the fraction of words that would have
to be inserted, deleted or replaced. The function was written for speech
transcripts and compares any two lists.

@chunk[<r13-measure>
(define (token-error-rate net source-vocab target-vocab pairs)
  (define hypotheses
    (translate net source-vocab target-vocab (map car pairs)))
  (define-values (errors words)
    (for/fold ([errors 0] [words 0])
              ([pair (in-list pairs)] [hypothesis (in-list hypotheses)])
      (define reference (string-split (cdr pair)))
      (values (+ errors (edit-distance reference (string-split hypothesis)))
              (+ words (length reference)))))
  (/ errors (exact->inexact words)))]

@chunk[<*>
  <r13-require>
  <r13-provide>
  <r13-encoder>
  <r13-attention>
  <r13-decoder>
  <r13-model>
  <r13-loss>
  <r13-step>
  <r13-device>
  <r13-run>
  <r13-train>
  <r13-translate>
  <r13-measure>]
