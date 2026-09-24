#lang scribble/manual

@(require (for-label (except-in racket/base
                                abs cos exp log sin sort sqrt max min length
                                + - * /)
                     torch))

@title[#:tag "examples" #:style 'toc]{Worked examples}

Each example is a literate program: the prose and the code are one file
under @filepath{examples/racket/}, and the chapters below are those files
rendered. Every one exports a @racket[run-example] thunk that a harness
under @filepath{examples/test/} drives from its @racket[main] submodule and
checks from its @racket[test] submodule, and most have a PyTorch twin under
@filepath{examples/python/} that the parity suite holds them to.

Run one from a checkout, or the whole suite:

@verbatim|{
racket examples/test/05-mnist.rkt
raco test examples/test/
}|

They build on one another. The first four are the primitives: seeded
draws, arithmetic, matrix products, and a gradient. The perceptron and the
MNIST convnet are the first models and the first training loops. The
character GPT, the LSTM and the translator are the sequence models; the
speech recogniser is the audio one; the diffusion model, the DCGAN and the
variational autoencoder are the generative ones; and the ResNet is the
classic supervised-vision recipe.

@margin-note{These chapters are woven from files outside the
@racketmodname[torch] package, so they render from a checkout or a
link-mode install; see @filepath{docs/building.md}.}

@include-section[(submod "../../../examples/racket/00-randn.rkt" doc)]
@include-section[(submod "../../../examples/racket/01-arith.rkt" doc)]
@include-section[(submod "../../../examples/racket/02-matmul.rkt" doc)]
@include-section[(submod "../../../examples/racket/03-autograd.rkt" doc)]
@include-section[(submod "../../../examples/racket/04-mlp.rkt" doc)]
@include-section[(submod "../../../examples/racket/05-mnist.rkt" doc)]
@include-section[(submod "../../../examples/racket/06-gpt.rkt" doc)]
@include-section[(submod "../../../examples/racket/07-asr.rkt" doc)]
@include-section[(submod "../../../examples/racket/08-diffusion.rkt" doc)]
@include-section[(submod "../../../examples/racket/09-resnet.rkt" doc)]
@include-section[(submod "../../../examples/racket/10-dcgan.rkt" doc)]
@include-section[(submod "../../../examples/racket/11-vae.rkt" doc)]
@include-section[(submod "../../../examples/racket/12-char-rnn.rkt" doc)]
@include-section[(submod "../../../examples/racket/13-translation.rkt" doc)]
