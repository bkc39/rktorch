#lang scribble/manual

@(require "common.rkt"
          (for-label racket/base
                     racket/contract
                     (only-in torch
                              backward! detach grad grad-enabled? has-grad?
                              maybe-grad mul native-collect-at-troughs
                              requires-grad! requires-grad? sum tensor tensor?
                              with-no-grad)
                     (only-in torch/nn zero-grads!)))

@title{Automatic differentiation}

@defmodule[torch #:link-target? #f]

Operations on a tensor marked with @racket[requires-grad!] are recorded, so
that @racket[backward!] can walk the record and accumulate derivatives into
the tensors that fed the result.

@section{Marking and reading}

@defproc[(requires-grad! [t tensor?] [on? boolean? #t]) tensor?]{
Marks @racket[t] as one to track, or stops tracking it when @racket[on?] is
@racket[#f]. Answers @racket[t].

@torch-examples[
(requires-grad? (requires-grad! (tensor '(2.0 3.0))))
]}

@defproc[(requires-grad? [t tensor?]) boolean?]{
Whether @racket[t] is tracked.}

@defproc[(grad [t tensor?]) tensor?]{
The gradient accumulated into @racket[t]. Raises if none has been
accumulated --- because no backward pass has run, or because @racket[t] is
not a tracked leaf. Use @racket[has-grad?] to ask first, or
@racket[maybe-grad] to get @racket[#f] instead of an exception.

@torch-examples[
(define x (requires-grad! (tensor '(2.0 3.0))))
(backward! (sum (mul x x)))
(grad x)
]}

@defproc[(has-grad? [t tensor?]) boolean?]{
Whether a gradient has been accumulated into @racket[t].}

@defproc[(maybe-grad [t tensor?]) (or/c tensor? #f)]{
As @racket[grad], answering @racket[#f] where @racket[grad] would raise.}

@section{The backward pass}

@defproc[(backward! [t tensor?]) void?]{
Differentiates the one-element tensor @racket[t] with respect to every
tracked tensor that contributed to it, accumulating each derivative into
that tensor's gradient.

Gradients @emph{accumulate} rather than replace, which is what makes
gradient accumulation across several batches possible --- and is why a
training loop clears them every step, with
@racket[zero-grads!].

The end of @racket[backward!] is the trough of a training step, so when
enough native memory has built up since the last one it runs a full
collection before returning, which can take a few hundred milliseconds.
@racket[native-collect-at-troughs] turns that off.}

@section{Turning tracking off}

@defform[(with-no-grad body ...+)]{
Evaluates @racket[body] with recording disabled, answering the last
result. Use it around evaluation, metrics, and in-place parameter updates
--- anywhere the result is not something to differentiate.

@torch-examples[
(grad-enabled?)
(with-no-grad (grad-enabled?))
]}

@defproc[(grad-enabled?) boolean?]{
Whether operations are currently being recorded.}

@defproc[(detach [t tensor?]) tensor?]{
A tensor sharing @racket[t]'s storage but carrying no history, so it is a
leaf of any later backward pass.}

@defproc[(copy! [t tensor?] [source tensor?]) void?]{
Writes @racket[source]'s values into @racket[t] in place, broadcasting
@racket[source] to @racket[t]'s shape and converting it to @racket[t]'s
dtype and device; @tt{t.copy_(source)}. Inside @racket[with-no-grad] it
updates a leaf that requires a gradient, such as the image a style
transfer optimises, without recording the write.

@torch-examples[
(define t (zeros 3))
(copy! t (tensor '(1.0 2.0 3.0)))
(tensor->list t)
]}
