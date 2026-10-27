#lang scribble/manual

@(require torch/scribblings/common
          torch/scribblings/labels
          (for-label torch/audio/data
                     torch/audio/functional
                     torch/audio/librispeech
                     torch/audio/metrics))

@title{Audio}

Waveforms in, spectrogram frames out, and the two error rates a speech
recogniser is judged by. The speech example under
@secref["ex-asr"] is the loop around these.

@section{Files}

@defmodule[torch/audio/data]

@defproc[(audio-info [path path-string?])
         (values exact-nonnegative-integer? exact-positive-integer?
                 exact-positive-integer?)]{
The frame count, sample rate and channel count of the audio file at
@racket[path], read from its header without decoding the samples.}

@defproc[(load-audio [path path-string?]
                     [#:frame-offset frame-offset exact-nonnegative-integer? 0]
                     [#:num-frames num-frames (or/c exact-nonnegative-integer? #f) #f])
         (values tensor? exact-positive-integer?)]{
The samples of the file at @racket[path] as a float32 @tt{[channels
frames]} tensor in @tt{[-1, 1]}, and its sample rate, as
@tt{torchaudio.load}. @racket[frame-offset] and @racket[num-frames] read a
window of the file.}

@section{Spectral features}

@defmodule[torch/audio/functional]

@defproc[(log-mel-spectrogram [samples tensor?]
                              [#:sample-rate sample-rate exact-positive-integer?]
                              [#:n-fft n-fft exact-positive-integer? 400]
                              [#:hop-length hop-length exact-positive-integer? 160]
                              [#:n-mels n-mels exact-positive-integer? 80]
                              [#:eps eps (and/c rational? (>=/c 0)) 1e-6]
                              [#:filterbank filterbank (or/c #f tensor?) #f])
         tensor?]{
The log of the mel-scaled power spectrogram of a @tt{[channels frames]}
waveform: a short-time Fourier transform over @racket[n-fft]-sample
windows every @racket[hop-length] samples, its power binned onto
@racket[n-mels] mel bands for @racket[sample-rate], and @racket[eps] added
before the logarithm. The defaults are the usual 25 ms window and 10 ms
hop at 16 kHz. Built where @racket[samples] live.

A given @racket[filterbank] replaces the @racket[mel-filterbank] the call
would build, and must be that tensor: @tt{[n-fft/2+1, n-mels]}, in the
dtype of @racket[samples] and on its device. A caller featurizing many
waveforms at one rate builds it once.}

@defproc[(mel-filterbank [#:n-freqs n-freqs exact-positive-integer?]
                         [#:n-mels n-mels exact-positive-integer?]
                         [#:sample-rate sample-rate exact-positive-integer?]
                         [#:f-min f-min (and/c rational? (>=/c 0)) 0.0]
                         [#:f-max f-max (or/c #f (and/c rational? positive?)) #f]
                         [#:device device (or/c #f device/c) #f]
                         [#:dtype dtype (or/c #f 'float32 'float64) #f])
         tensor?]{
The @tt{[n-freqs, n-mels]} triangular HTK mel filters,
@tt{torchaudio.functional.melscale_fbanks} with @tt{mel_scale="htk"} and
@tt{norm=None}; @racket[f-max] defaults to half the rate.}

@defproc[(edit-distance [reference list?] [hypothesis list?])
         exact-nonnegative-integer?]{
The Levenshtein distance between two sequences: the least number of
insertions, deletions and substitutions turning one into the other. The
metrics below apply it to words and to characters.}

@section{Error rates}

@defmodule[torch/audio/metrics]

@deftogether[(@defproc[(wer [reference string?] [hypothesis string?])
                       (and/c rational? (>=/c 0))]
              @defproc[(cer [reference string?] [hypothesis string?])
                       (and/c rational? (>=/c 0))])]{
The word and character error rates of @racket[hypothesis] against a
non-empty @racket[reference]: the @racket[edit-distance] over words, or
over characters, divided by the reference's length. Exact rationals, so
they can be summed over a test set without drift.}

@section{LibriSpeech}

@defmodule[torch/audio/librispeech]

The read-speech corpus, by split. An archive is fetched once into the
cache directory and extracted; the utterances are its FLAC files with
their transcripts. Downloaded on demand, not distributed by rktorch: check
its license before use.

@defproc[(librispeech-utterances [split (or/c "dev-clean" "test-clean")])
         (listof utterance?)]{
Every utterance of @racket[split], sorted by id, fetching and extracting
the split's archive the first time.}

@deftogether[(@defproc[(utterance? [v any/c]) boolean?]
              @defproc[(utterance-id [u utterance?]) string?]
              @defproc[(utterance-path [u utterance?]) path?]
              @defproc[(utterance-transcript [u utterance?]) string?])]{
An utterance: its LibriSpeech id, the path of its FLAC file, and its
transcript in the corpus's upper-case form.}

@defproc[(load-utterance [u utterance?]) (values tensor? exact-positive-integer?)]{
@racket[load-audio] on the utterance's file: its samples and sample rate.}

@defproc[(load-librispeech-fixture)
         (values tensor? exact-positive-integer? string?)]{
One committed utterance --- samples, sample rate and transcript --- for
the tests and the offline example, so neither fetches the corpus.}

@defproc[(librispeech-fixture-utterance) utterance?]{
The committed utterance as an @racket[utterance?] value: id
@tt{1272-128104-0000}, its FLAC file and its transcript.}

@defproc[(librispeech-root [split (or/c "dev-clean" "test-clean")]) path?]{
The directory holding @filepath{LibriSpeech/<split>}, the
@tt{root} @tt{torchaudio.datasets.LIBRISPEECH} takes, fetching and
extracting the split's archive the first time.}

@defproc[(librispeech [utterances (listof utterance?)]
                      [#:transform transform
                                   (-> tensor? exact-positive-integer?
                                       tensor?)
                                   (lambda (samples rate) samples)])
         librispeech?]{
A dataset over @racket[utterances], in their order. Item @racket[i] is
two values: @racket[transform] applied to the utterance's samples and
sample rate, and its transcript. With the default @racket[transform] it
is @tt{LIBRISPEECH(root, url=split)[i]}'s waveform and transcript, a
float32 @tt{[1, T]} tensor and a string. @racket[transform]'s result has
at least one dimension, the last being time.

Nothing is cached: each item is decoded from its file when it is asked
for, so the dataset holds no samples and loader workers may build batches
concurrently, provided @racket[transform] is safe to call from several
threads.

Under @racket[default-collate], a batch is three values: the items
zero-padded at the end of their last dimension to the longest and
stacked to @tt{[B, d..., T-max]}, the list of their unpadded last
dimensions, and the list of their transcripts, all in index order. Items
that differ in rank, in a leading dimension, in dtype or in device raise
@racket[exn:fail:contract]. Any other collate receives
@racket[(list features transcript)] per index.

@torch-examples[
(require torch/audio/librispeech torch/data/loader)
(define ds (librispeech (list (librispeech-fixture-utterance))))
(define-values (samples transcript) (dataset-ref ds 0))
(shape samples)
(define-values (x lengths transcripts)
  (dataset-batch ds '(0 0) default-collate))
(shape x)
lengths
]}

@defproc[(librispeech? [v any/c]) boolean?]{
The predicate.}
