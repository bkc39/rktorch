#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     (only-in torch tensor?)
                     (only-in torch/data/loader dataloader tensor-dataset)
                     (only-in torch/nn nll-loss)
                     torch/data/translation))

@title{Translation pairs}

@defmodule[torch/data/translation]

The English-French sentence pairs of the PyTorch seq2seq tutorial, prepared as
the tutorial prepares them: each sentence lower-cased and reduced to ASCII
words, the pairs filtered to short sentences that open with a pronoun and a
form of @italic{to be}, a word-level vocabulary for each language. Of the
file's 135,842 lines 11,445 pairs survive, over 4,599 French and 2,989
English words, the tutorial's own counts.

A pair is @racket[(cons source target)], both normalized strings. The source
language is French unless @racket[#:source] says otherwise, so the default
task is translating into English, as in the tutorial.

@defproc[(load-translation-pairs
          [#:source source (or/c 'eng 'fra) 'fra]
          [#:max-length max-length exact-positive-integer? 10]
          [#:prefixes prefixes (or/c (listof string?) #f) tutorial-prefixes])
         (listof (cons/c string? string?))]{
Downloads the tutorial's archive (3 MB) on first use, caches it under
@envvar{RKTORCH_TRANSLATION_DIR} or the system cache directory, and parses
@filepath{data/eng-fra.txt} out of it with @racket[parse-pairs]. A response
that fails @racket[translation-archive?], an error page or a download cut
short, is not cached.
}

@defproc[(translation-archive? [path path-string?]) boolean?]{
Whether the file at @racket[path] is a complete zip archive holding
@filepath{data/eng-fra.txt}.
}

@defproc[(load-translation-fixture [#:source source (or/c 'eng 'fra) 'fra])
         (listof (cons/c string? string?))]{
The 287 pairs of the committed excerpt, for tests and offline examples: every
fortieth surviving pair of the full file, so its sentences span the file's
range of lengths. The excerpt also carries lines the filter rejects.
}

@defproc[(parse-pairs [text string?]
                      [#:source source (or/c 'eng 'fra) 'fra]
                      [#:max-length max-length exact-positive-integer? 10]
                      [#:prefixes prefixes (or/c (listof string?) #f)
                                  tutorial-prefixes])
         (listof (cons/c string? string?))]{
Parses tab-separated @tt{English<TAB>French} lines; columns after the second
and lines without a tab are ignored. A pair is kept when both normalized
sentences have fewer than @racket[max-length] words and the English one starts
with one of @racket[prefixes]; @racket[#f] keeps every opening.
}

@defproc[(normalize-sentence [s string?]) string?]{
The tutorial's @tt{normalizeString}: trimmed, lower-cased, accents removed
with @racket[strip-accents], @litchar{!} and @litchar{?} split off as words of
their own, and every other run of non-letters, the full stop included,
collapsed to one space.

@racketblock[
(normalize-sentence "Qu'est-il arrivé ?")
(code:comment "=> \"qu est il arrive ?\"")
]
}

@defproc[(strip-accents [s string?]) string?]{
Decomposes @racket[s] (NFD) and drops the combining marks.
}

@defthing[tutorial-prefixes (listof string?)]{
The English openings the tutorial keeps: @tt{"i am "}, @tt{"i m "},
@tt{"he is"}, @tt{"he s "} and their @italic{she}, @italic{you}, @italic{we}
and @italic{they} counterparts.
}

@section{Vocabularies}

@deftogether[(@defthing[pad-id exact-nonnegative-integer?]
              @defthing[sos-id exact-nonnegative-integer?]
              @defthing[eos-id exact-nonnegative-integer?])]{
The ids of the three special words every vocabulary starts with:
@tt{<pad>} is 0, @tt{<sos>} 1, @tt{<eos>} 2. The tutorial has no padding word
and pads with its start-of-sentence id; a separate one lets
@racket[nll-loss] skip padded positions through @racket[#:ignore-index].
}

@defproc[(words->vocab [sentences (listof string?)]) word-vocab?]{
A vocabulary of the specials followed by each word of @racket[sentences] in
order of first appearance.
}

@defproc[(pairs->vocabs [pairs (listof (cons/c string? string?))])
         (values word-vocab? word-vocab?)]{
The source and the target vocabulary of @racket[pairs].
}

@deftogether[(@defproc[(word-vocab? [v any/c]) boolean?]
              @defproc[(word-vocab-words [v word-vocab?]) (vectorof string?)]
              @defproc[(vocab-size [v word-vocab?]) exact-positive-integer?])]{
The predicate, the words by id, and their count, specials included.
}

@defproc[(encode-sentence [v word-vocab?] [sentence string?])
         (listof exact-nonnegative-integer?)]{
The ids of a normalized sentence's words followed by @racket[eos-id]. A word
outside the vocabulary is an error.
}

@defproc[(decode-tokens [v word-vocab?]
                        [ids (or/c tensor? (listof exact-nonnegative-integer?))])
         string?]{
The words of @racket[ids] up to the first @racket[eos-id], skipping
@racket[pad-id] and @racket[sos-id].
}

@deftogether[(@defproc[(sentences->tensor
                        [v word-vocab?]
                        [sentences (and/c (listof string?) pair?)]
                        [#:width width (or/c exact-positive-integer? #f) #f])
                       tensor?]
              @defproc[(pairs->tensors
                        [pairs (and/c (listof (cons/c string? string?)) pair?)]
                        [source-vocab word-vocab?]
                        [target-vocab word-vocab?]
                        [#:width width (or/c exact-positive-integer? #f) #f])
                       (values tensor? tensor?)])]{
Encoded sentences as the rows of an @racket['int64] tensor, each ending in
@racket[eos-id] and padded with @racket[pad-id] to @racket[width], or to the
longest row when it is @racket[#f]. With the default @racket[#:max-length] a
sentence has at most nine words, so a width of 10 fits every row. The two
tensors of @racket[pairs->tensors] go straight into a
@racket[tensor-dataset] for a @racket[dataloader].
}
