#lang scribble/manual

@(require torch/scribblings/labels)

@title[#:tag "examples" #:style 'toc]{Worked examples}

Each example is a literate program: the prose and the code are one file
under @filepath{torch/examples/}, and the chapters below are those files
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
variational autoencoder are the generative ones; the ResNet is the
classic supervised-vision recipe; and the last three start from
pretrained weights: classifying photographs with ImageNet's ResNet-18,
fine-tuning it on ants and bees, and painting a photograph in the style of
another with VGG-16's features.

@include-section[(submod torch/examples/00-randn doc)]
@include-section[(submod torch/examples/01-arith doc)]
@include-section[(submod torch/examples/02-matmul doc)]
@include-section[(submod torch/examples/03-autograd doc)]
@include-section[(submod torch/examples/04-mlp doc)]
@include-section[(submod torch/examples/05-mnist doc)]
@include-section[(submod torch/examples/06-gpt doc)]
@include-section[(submod torch/examples/07-asr doc)]
@include-section[(submod torch/examples/08-diffusion doc)]
@include-section[(submod torch/examples/09-resnet doc)]
@include-section[(submod torch/examples/10-dcgan doc)]
@include-section[(submod torch/examples/11-vae doc)]
@include-section[(submod torch/examples/12-char-rnn doc)]
@include-section[(submod torch/examples/13-translation doc)]
@include-section[(submod torch/examples/14-imagenet doc)]
@include-section[(submod torch/examples/15-finetune doc)]
@include-section[(submod torch/examples/16-style-transfer doc)]
