#lang scribble/manual

@(require "common.rkt"
          (for-label racket/base
                     racket/contract
                     (only-in torch
                              backward! detach grad grad-enabled? has-grad?
                              maybe-grad mul requires-grad! requires-grad?
                              sum tensor tensor? with-no-grad)
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
@racket[zero-grads!].}

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
