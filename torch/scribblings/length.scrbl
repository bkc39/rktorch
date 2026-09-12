#lang scribble/manual

@(require (for-label (except-in racket/base length)
                     racket/contract
                     (only-in torch gen:sized length sized? tensor? zeros)
                     torch/data/loader))

@title{Length}

@defmodule[torch #:link-target? #f]

@defproc[(length [v sized?]) exact-nonnegative-integer?]{
Python's @tt{len}, shadowing @racketmodname[racket/base]'s
@racket[length] the way @racket[+] is shadowed: a list, vector, string or
hash answers what it always did, a tensor answers its first dimension,
and a dataset or loader answers its number of items or batches.

@racketblock[
(length '(1 2 3))
(length (zeros 4 2))
(length (dataloader ds #:batch-size 64))
]

A rank-zero tensor has no length, as @tt{len} of a 0-d tensor raises.
}

@defproc[(sized? [v any/c]) boolean?]{
Recognises anything @racket[length] accepts.
}

@defthing[gen:sized any/c]{
The generic behind @racket[length], with that one method. A structure
implements it with @racket[#:methods]; @racket[define-dataset] does so for
every dataset.
}
