#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     (only-in torch backward! tensor? to zeros)
                     (only-in torch/nn
                              GRU LSTM adam clip-grad-norm! eval! gru? load-state!
                              lstm? named-parameters parameters step!
                              zero-grads!)))

@title{Recurrent layers}

@defmodule[torch/nn #:link-target? #f]

@deftogether[(@defproc[(LSTM [input-size exact-positive-integer?]
                             [hidden-size exact-positive-integer?]
                             [#:num-layers num-layers exact-positive-integer? 1]
                             [#:bias? bias? boolean? #t]
                             [#:batch-first? batch-first? boolean? #f]
                             [#:dropout dropout (and/c real? (>=/c 0) (<=/c 1)) 0.0]
                             [#:bidirectional? bidirectional? boolean? #f])
                       lstm?]
              @defproc[(GRU [input-size exact-positive-integer?]
                            [hidden-size exact-positive-integer?]
                            [#:num-layers num-layers exact-positive-integer? 1]
                            [#:bias? bias? boolean? #t]
                            [#:batch-first? batch-first? boolean? #f]
                            [#:dropout dropout (and/c real? (>=/c 0) (<=/c 1)) 0.0]
                            [#:bidirectional? bidirectional? boolean? #f])
                       gru?])]{
Multi-layer recurrences over a whole sequence in one call, PyTorch's
@tt{nn.LSTM} and @tt{nn.GRU}, run by cudnn on CUDA. The input has rank
three: @tt{[T, N, input-size]}, or @tt{[N, T, input-size]} when
@racket[batch-first?]. Applying a layer answers the output and the final
state as values:

@racketblock[
(define-values (output h-n c-n) (lstm x))
(define-values (output h-n) (gru x))
]

@racket[output] holds the top layer's hidden state at every step,
@tt{hidden-size} wide, or twice that when @racket[bidirectional?]. Each
state has shape @tt{[num-layers × directions, N, hidden-size]} whatever
@racket[batch-first?] says. The initial state is zero unless it follows
the input, @racket[(lstm x h-0 c-0)] or @racket[(gru x h-0)]; feeding one
call's final state to the next continues the sequence, which is how a
decoder steps one token at a time.

@racket[dropout] applies between layers, never after the last, and only in
training mode; @racket[eval!] turns it off.

The parameters carry PyTorch's names and order, @tt{weight_ih_l0},
@tt{weight_hh_l0}, @tt{bias_ih_l0}, @tt{bias_hh_l0}, then
@tt{_l1} and so on, each direction's @tt{_reverse} twins following it, so
@racket[load-state!] reads a checkpoint written from @tt{nn.LSTM}. They
are drawn as @tt{reset_parameters} draws them, uniform within
@tt{1/sqrt(hidden-size)}: under one seed a layer starts from PyTorch's
values.

After a move with @racket[to], the first call on a CUDA device packs the
weights into the single buffer cudnn wants; the parameters stay the same
tensors, so an optimizer built before or after sees them alike.

Unbatched rank-two inputs, projections (@tt{proj_size}) and packed
sequences are not supported.
}

@deftogether[(@defproc[(lstm? [v any/c]) boolean?]
              @defproc[(gru? [v any/c]) boolean?])]{
Whether @racket[v] was built by @racket[LSTM] or by @racket[GRU].
}

@defproc[(clip-grad-norm! [params (listof tensor?)]
                          [max-norm (and/c real? (>=/c 0))])
         tensor?]{
Scales the gradients of @racket[params] in place so that their joint L2
norm, taken over all of them as one vector, is at most
@racket[max-norm], and answers the norm they had before, as PyTorch's
@tt{clip_grad_norm_}. Gradients already within the bound are left as they
are, and parameters without a gradient are skipped; a bound of zero zeroes
every gradient. It belongs between
the backward pass and the optimizer step:

@racketblock[
(zero-grads! opt)
(backward! loss)
(clip-grad-norm! (parameters net) 5.0)
(step! opt)
]

Recurrent networks need it because a gradient carried back through many
steps can grow without bound. The scale never leaves the device, so
clipping does not wait on the GPU; reading the returned norm does.
}
