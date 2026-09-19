<div align="center">

# <img src="docs/images/rktorch-logo.png" alt="" height="106" hspace="8" align="absmiddle"> rktorch

**Tensors, autograd, and neural networks for Racket — à la [PyTorch](https://pytorch.org/)**

[![Build](https://img.shields.io/github/actions/workflow/status/bkc39/rktorch/nix.yml?label=build)](https://github.com/bkc39/rktorch/actions/workflows/nix.yml)
[![Docs](https://img.shields.io/badge/docs-racket--lang.org-blue.svg)](https://docs.racket-lang.org/torch)
[![Package](https://img.shields.io/badge/raco%20pkg-torch-purple.svg)](https://pkgs.racket-lang.org/package/torch)
[![License](https://img.shields.io/badge/license-Apache--2.0%20AND%20CC--BY--4.0%20AND%20CC--BY--2.0--FR-blue.svg)](#license)

</div>

> [!WARNING]
> **This library is a work in progress.** The API is still changing and the
> package is not yet on the Racket catalog. Expect breaking changes.

`rktorch` provides Racket bindings to
[libtorch](https://docs.pytorch.org/cppdocs/), the C++ library behind PyTorch.
It aims to provide a PyTorch-like API for manipulating tensors and building
neural networks: GPU-accelerated tensor computations, tape-based automatic
differentiation, a `define-layer` form for models, and PyTorch-style datasets
and loaders.

The bindings are checked against PyTorch itself: seeded draws, initializers,
optimizers, and whole training runs are compared with Python twins, so a model
ported from PyTorch trains the same way here. CPU is supported everywhere, CUDA
on Linux, and MPS on Apple Silicon.

## Install

> [!NOTE]
> `raco pkg install torch` is forthcoming. The package goes on the Racket
> catalog once the library is ready for a release; that work is tracked in
> [#149](https://github.com/bkc39/rktorch/issues/149).

Until then, build from a checkout with [Nix](https://nixos.org/). The first
`nix develop` builds `libtorchrkt`, the small C++ shim linked against libtorch,
and installs the Racket package and its dependencies into the checkout:

```sh
git clone https://github.com/bkc39/rktorch.git
cd rktorch
nix develop
racket -ie "(require torch)"
```

[Building from source](docs/building.md) covers the
[Nix build](docs/building.md#build-and-test) and the
[development loop](docs/building.md#development-shells): the CUDA and OCaml
shells, running the tests, and re-staging the native library after a C++ change.

## Usage

![A Racket REPL session: a seeded random tensor, a matrix product, a byte
string as a uint8 tensor, a gradient from backward!, and a Linear layer applied
to a tensor](docs/images/repl.gif)

A tensor prints in the Racket REPL exactly as it does in Python:

```racket
(require torch)

(manual-seed! 0)
(define a (randn 2 3))
a
```

```text
tensor([[ 1.5410, -0.2934, -2.1788],
        [ 0.5684, -1.0845, -1.3986]])
```

`+ - * /` work on tensors and numbers alike, `@` is matrix multiplication, and
`T` is the transpose, as in `a @ a.T`:

```racket
(define b (tensor '((1 2 3) (4 5 6))))
(+ a b)
(@ a (T a))
```

```text
tensor([[2.5410, 1.7066, 0.8212],
        [4.5684, 3.9155, 4.6014]])
tensor([[7.2079, 4.2414],
        [4.2414, 3.4554]])
```

Data comes in from lists, vectors, homogeneous vectors, and byte strings:

```racket
(tensor #"\0\1\2\377")
```

```text
tensor([  0,   1,   2, 255], dtype=torch.uint8)
```

### Automatic differentiation

```racket
(define x (tensor '(1.0 2.0 3.0) #:requires-grad? #t))
(backward! (~> x (* x) Σ))
(grad x)
```

```text
tensor([2., 4., 6.])
```

### Neural networks

`define-layer` is the `nn.Module` analog. `#:init` is the constructor body;
fields holding layers or parameters are registered by what they hold, and an
instance applies as a procedure.

```racket
(require torch torch/nn torch/data/loader
         (only-in torch/data/mnist load-mnist-fixture))

(define-layer mlp (flat fc1 fc2)
  #:init (hidden)
  (set! flat (Flatten))
  (set! fc1 (Linear 784 hidden))
  (set! fc2 (Linear hidden 10))
  #:forward (x)
  (~> x flat fc1 relu fc2))

(define-values (xs ys) (load-mnist-fixture))   ; 256 digits shipped with the package
(manual-seed! 0)
(define net (mlp 64))
(define opt (adam (parameters net) #:lr 0.01))
(define loader
  (dataloader (tensor-dataset xs ys)
              #:batch-size 32 #:shuffle? #t
              #:generator (make-generator 0)))

(define (accuracy)
  (with-no-grad
    (item (mean (to-dtype (eq (argmax (net xs) 1) ys) 'float32)))))

(for ([epoch (in-range 5)])
  (for ([(xb yb) (in-dataloader loader)])
    (zero-grads! opt)
    (define loss (cross-entropy (net xb) yb))
    (backward! loss)
    (step! opt))
  (printf "epoch ~a  accuracy ~a\n" epoch (accuracy)))
```

```text
epoch 0  accuracy 0.78515625
epoch 1  accuracy 0.9375
epoch 2  accuracy 0.93359375
epoch 3  accuracy 0.9765625
epoch 4  accuracy 1.0
```

Parameters are named as PyTorch names them, so checkpoints read naturally:

```racket
(map car (named-parameters net))
;; => '("fc1.weight" "fc1.bias" "fc2.weight" "fc2.bias")
```

### Devices

```racket
(define dev (accelerator-if-available))   ; cuda, mps, or cpu
(to net dev)                              ; moves a layer in place, like nn.Module.to
(net (to xs dev))
(randn 2 2 #:device dev)
```

## Examples

The examples are literate programs, each with a PyTorch twin it is tested
against:

- [Seeded draws](examples/racket/00-randn.rkt), [arithmetic](examples/racket/01-arith.rkt),
  and [matrix products](examples/racket/02-matmul.rkt)
- [Autograd](examples/racket/03-autograd.rkt)
- [A two-layer perceptron](examples/racket/04-mlp.rkt)
- [A convolutional network on MNIST](examples/racket/05-mnist.rkt), trained on
  shuffled minibatches from a dataloader
- [A character-level GPT](examples/racket/06-gpt.rkt), with
  [training](scripts/train-gpt.rkt) and [generation](scripts/generate-gpt.rkt) scripts
- [Speech recognition on LibriSpeech](examples/racket/07-asr.rkt): a spectral
  front end, a transformer encoder-decoder, and CTC, with a
  [training script](scripts/train-asr.rkt)
- [Image generation on CIFAR-10](examples/racket/08-diffusion.rkt): a
  class-conditional DDPM with a UNet, trained with an EMA of the weights
- [A character-level LSTM](examples/racket/12-char-rnn.rkt): gradient
  clipping, and sampling at a temperature from a carried state

## Documentation

- The reference manual is written in Scribble under
  [`torch/scribblings`](torch/scribblings) and will be served at
  [docs.racket-lang.org/torch](https://docs.racket-lang.org/torch) once the
  package is on the catalog.
- [Building from source](docs/building.md): the Nix flake, development shells,
  accelerators, and the libtorch source knob.
- [`AGENTS.md`](AGENTS.md): the layout of the repository and its conventions.
- Roadmap: the [open epics](https://github.com/bkc39/rktorch/issues?q=is%3Aissue+is%3Aopen+epic+in%3Atitle)
  on the issue tracker; [`plans/v0-scaffold.md`](plans/v0-scaffold.md) records
  the original scope.

## Internals

A hand-written `extern "C"` shim over libtorch carries the core of the library,
and the wider ATen surface is generated from PyTorch's operator schema. Tensors
are handles owned by Racket's garbage collector, with a ledger that reports
native memory to it. Read [internals.md](docs/internals.md) for how memory is
managed across the FFI boundary and
[v1-codegen-nn.md](docs/design/v1-codegen-nn.md) for the code generator and the
design of the `nn` layer.

## Acknowledgements

Many thanks to the PyTorch team for libtorch, and to
[@LaurentMazare](https://github.com/LaurentMazare) and Jane Street for
[ocaml-torch](https://github.com/janestreet/torch), whose design this library
follows and which it keeps as a reference implementation.

## License

`Apache-2.0 AND CC-BY-4.0 AND CC-BY-2.0-FR`, as declared in
[`torch/info.rkt`](torch/info.rkt). The code is Apache-2.0; the committed
data fixtures carry the licenses of their sources, recorded in the `NOTICE`
beside each one ([`torch/audio/fixtures`](torch/audio/fixtures/NOTICE),
[`torch/data/fixtures`](torch/data/fixtures/NOTICE)).
