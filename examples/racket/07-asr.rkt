#lang scribble/lp2

@(require (for-label (except-in racket/base abs cos exp log sin sqrt max min + - * /)
                     torch torch/nn))

@section[#:tag "ex-asr"]{Speech to text on LibriSpeech with CTC}

The speech capstone: a character-level connectionist temporal classification
(CTC) recognizer over LibriSpeech utterances. The whole arc meets here ---
FLAC decode (@racket[load-utterance]), the @racket[log-mel-spectrogram]
front-end, a strided @racket[Conv1d] encoder, @racket[ctc-loss] training,
greedy decoding, and @racket[wer]/@racket[cer] scoring.

CTC is what makes speech tractable without alignments: the model emits a
distribution over @tt{vocab + blank} at every downsampled frame, and the loss
marginalizes over @emph{every} monotonic alignment between those frames and
the reference characters. No one ever labels which frame the Q of QUILTER
lands on; the loss sums the paths and the blank symbol soaks up the silences
and stretched phonemes.

@chunk[<r07-require>
(require torch torch/nn
         (only-in torch/audio/functional log-mel-spectrogram)
         (only-in torch/audio/librispeech
                  librispeech-utterances load-librispeech-fixture
                  load-utterance utterance-transcript)
         (only-in torch/audio/metrics cer wer)
         (only-in torch/data/text decode encode text->vocab))]

@chunk[<r07-provide>
(provide asr pick-device
         run-example greedy-decode utterance-features train-librispeech)]

@bold{The model.} Two @racket[Conv1d] layers stride over the 80-mel frames,
each halving time (kernel 3, stride 2, padding 1), so the head classifies
4x-downsampled frames --- roughly 40ms of speech each --- into
@tt{vocab-size + 1} classes. The extra class is the CTC blank, indexed
@emph{after} the real characters so character ids pass through unshifted.
The forward returns @racket[log-softmax]ed frames as @tt{[B, T', C]}; the
loss and the decoder both consume that one output.

@chunk[<r07-model>
(define-module asr (n-mels vocab-size #:n-hidden [n-hidden 64])
  #:submodules ([conv1 (Conv1d n-mels n-hidden 3 #:stride 2 #:padding 1)]
                [conv2 (Conv1d n-hidden n-hidden 3 #:stride 2 #:padding 1)]
                [head (Linear n-hidden (add1 vocab-size))])
  #:forward (x)
  (with-default-device (tensor-device x)
    (define frames (relu (conv2 (relu (conv1 x)))))
    (log-softmax (head (t frames 1 2)) 2)))]

@bold{The device.} As in the earlier capstones.

@chunk[<r07-device>
(define (pick-device)
  (accelerator-if-available))]

@bold{Features.} One utterance becomes a @tt{[1, 80, T]} batch: channel 0 of
the decoded samples through the log-mel front-end, plus the batch dimension.

@chunk[<r07-features>
(define (utterance-features samples rate)
  (unsqueeze (log-mel-spectrogram (ref samples 0) #:sample-rate rate) 0))]

@bold{The deterministic core.} @racket[run-example] is the seeded, offline
entry the test harness and the PyTorch parity twin both drive: the committed
MISTER QUILTER fixture (90 characters, ~1.5s) trains a fixture-scale
@racket[asr] for @racket[steps] @racket[adam] steps. The vocabulary comes
from the transcript itself; the blank index is @racket[(vector-length vocab)].
@racket[ctc-loss] wants log-probs time-major --- @tt{[T', B, C]} --- so the
forward's @tt{[B, T', C]} output transposes once at the loss. The input
length is the model's own downsampled frame count, read off the output shape;
the target length is the transcript's character count.

@chunk[<r07-run>
(define (run-example #:steps [steps 5] #:device [device (pick-device)])
  (with-default-device device
    (manual-seed! 0)
    (define-values (samples rate transcript) (load-librispeech-fixture))
    (define vocab (text->vocab transcript))
    (define v-size (vector-length vocab))
    (define x (to-device (utterance-features samples rate) device))
    (define targets (unsqueeze (encode vocab transcript) 0))
    (define target-length (string-length transcript))
    (define net (asr 80 v-size))
    (define opt (adam (parameters net) #:lr 0.001))
    (define losses
      (for/list ([_ (in-range steps)])
        (zero-grads! opt)
        (define out (net x))
        (define loss
          (ctc-loss (t out 0 1) targets
                    #:input-lengths (list (cadr (tensor-shape out)))
                    #:target-lengths (list target-length)
                    #:blank v-size))
        (backward! loss)
        (step! opt)
        (item loss)))
    (values losses net vocab device)))]

@bold{Greedy decoding.} The simplest CTC decoder: argmax each frame, collapse
consecutive repeats, drop blanks, map what survives back through the
vocabulary. Collapsing @emph{before} dropping is what lets a blank separate a
genuine double letter --- @tt{L-blank-L} survives as LL where @tt{L-L}
collapses to L.

@chunk[<r07-decode>
(define (greedy-decode net vocab features)
  (define blank (vector-length vocab))
  ;; any parameter's device works: a module's tensors are colocated
  (define x (to-device features (tensor-device (car (parameters net)))))
  (define ids
    (in-eval-mode net
      (with-no-grad
        (map inexact->exact (tensor->list (argmax (net x) 2))))))
  (define kept
    (for/fold ([prev #f] [acc '()] #:result (reverse acc))
              ([id (in-list ids)])
      (values id
              (if (or (equal? id prev) (= id blank))
                  acc
                  (cons id acc)))))
  (decode vocab kept))]

@bold{The real thing.} @racket[train-librispeech] downloads the dev-clean
split (~337MB archive, cached under @envvar{RKTORCH_AUDIO_DIR} or the system
cache dir) and runs utterance-at-a-time epochs over the first @racket[limit]
utterances, building one vocabulary over their transcripts up front. Batch
size stays 1: LibriSpeech utterances vary from 2 to 30+ seconds, and padding
them into rectangles buys throughput at the cost of obscuring the CTC
mechanics this example exists to show. Each utterance's features are computed
on the fly on the training device. Loss prints as a per-epoch mean.

@chunk[<r07-train>
(define (train-librispeech #:epochs [epochs 10] #:limit [limit 64]
                           #:n-hidden [n-hidden 128]
                           #:device [device (pick-device)]
                           #:log-every [log-every 1])
  (with-default-device device
    (manual-seed! 0)
    (define utts (let ([all (librispeech-utterances "dev-clean")])
                   (if (< limit (length all))
                       (for/list ([u (in-list all)] [_ (in-range limit)]) u)
                       all)))
    (define vocab
      (text->vocab (apply string-append
                          (map utterance-transcript utts))))
    (define v-size (vector-length vocab))
    (define net (asr 80 v-size #:n-hidden n-hidden))
    (define opt (adam (parameters net) #:lr 0.001))
    (for ([epoch (in-range 1 (add1 epochs))])
      (define-values (total steps)
        (for/fold ([total 0.0] [steps 0])
                  ([u (in-list utts)])
          (define-values (samples rate) (load-utterance u))
          (define transcript (utterance-transcript u))
          (zero-grads! opt)
          (define out (net (to-device (utterance-features samples rate)
                                      device)))
          (define loss
            (ctc-loss (t out 0 1)
                      (unsqueeze (encode vocab transcript) 0)
                      #:input-lengths (list (cadr (tensor-shape out)))
                      #:target-lengths (list (string-length transcript))
                      #:blank v-size))
          (backward! loss)
          (step! opt)
          (values (+ total (item loss)) (add1 steps))))
      (when (zero? (modulo epoch log-every))
        (printf "epoch ~a/~a: mean loss ~a\n" epoch epochs (/ total steps))))
    (values net vocab)))]

@bold{Scoring.} Decode an utterance and hold it against its reference with
@racket[wer]/@racket[cer] --- the rates are exact rationals, so a report like
@tt{3/10} reads as literally three word edits over a ten-word reference:

@racketblock[
(define-values (net vocab) (train-librispeech))
(define-values (samples rate transcript) (load-librispeech-fixture))
(define hypothesis (greedy-decode net vocab (utterance-features samples rate)))
(wer transcript hypothesis)
(cer transcript hypothesis)
]

@chunk[<*>
<r07-require>
<r07-provide>
<r07-model>
<r07-device>
<r07-features>
<r07-run>
<r07-decode>
<r07-train>]
