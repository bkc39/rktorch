#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     (only-in torch tensor tensor? with-default-device)
                     (only-in torch/nn ctc-loss)
                     torch/audio/data
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
                              [#:eps eps (and/c rational? (>=/c 0)) 1e-6])
         tensor?]{
The log of the mel-scaled power spectrogram of a @tt{[channels frames]}
waveform: a short-time Fourier transform over @racket[n-fft]-sample
windows every @racket[hop-length] samples, its power binned onto
@racket[n-mels] mel bands for @racket[sample-rate], and @racket[eps] added
before the logarithm. The defaults are the usual 25 ms window and 10 ms
hop at 16 kHz. Built where @racket[samples] live.}

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
their transcripts.

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
