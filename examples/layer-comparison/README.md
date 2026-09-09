# Layer composition in Racket, Python, and OCaml

Open [side-by-side.html](side-by-side.html) for corresponding definitions in
three columns. Regenerate it after editing the sources:

```sh
nix develop --command python3 examples/layer-comparison/render.py
```

The ResNet components appear first: `Projection`, `ChannelNorm`,
`ResidualBlock`, `ResidualStage`, and `SmallResNet`. The remaining components
build causal self-attention, a feed-forward layer, and a transformer stack.

| Concern | Racket | Python | Jane Street OCaml |
|---|---|---|---|
| Construction | `define-layer` with `#:init` | `nn.Module.__init__` | A function receiving `Var_store.t` |
| Composition | `~>` pipelines | Nested calls or intermediate bindings | `\|>` pipelines |
| Parameter ownership | Registered layer tree | Registered module tree | Explicit variable store |
| Repeated children | `LayerList` and `in-layers` | `ModuleList` | Ordinary list |
| Matrix transpose | `(T weight)` | `weight.T` | `Tensor.transpose weight ~dim0:0 ~dim1:1` |
| Evaluation | `(eval! net)` then `(net x)` | `net.eval()` then `net(x)` | `Layer.forward_ net x ~is_training:false` |

Racket also provides `procedure->Layer`, with optional `#:parameters`,
`#:buffers`, and `#:children` association lists for explicit registration.
The three-column examples retain `define-layer` to compare its syntax.
The corresponding closure-based composition is:

```racket
(define projection (Projection 32 32))
(define drop (Dropout #:p 0.1))
(define block
  (procedure->Layer
   (lambda~> projection relu drop)
   #:children (list (cons "projection" projection)
                    (cons "drop" drop))))
```

Use `(require torch torch/nn "models.rkt")` for this snippet.
Registered children participate in parameter traversal and train/eval changes.
Omitting registration creates a layer with no registered captures.
OCaml's `Layer.of_fn` wraps a closure after its parameters have been registered
in the supplied store; it does not discover captured parameters.

The Racket forward bodies use threading for linear computations, with named
bindings where attention branches. `T` reverses every axis, so it is used on
the rank-two projection weight. Batched attention keys still use `transpose` to
swap only their last two axes.

## Architecture

The CNN uses a 3x3 stem, three stages at 16/32/64 channels with two blocks each,
global spatial averaging, and a classifier. The first block of each widening
stage downsamples by two. Each residual block explicitly computes an identity
or learned projection shortcut and adds it to the main path.

This is a ResNet-style comparison using channel-wise LayerNorm and biased
convolutions, not canonical ResNet-18. BatchNorm's running statistics and mode
handling are a further test of the layer API. The Racket spatial average uses
`mean-dim` from `torch/generated`.

The transformer accepts token representations of shape `[B,T,32]`, uses four
heads and two pre-norm blocks, and returns the same shape. Q/K/V projections,
scaled scores, the causal mask, softmax, head joining, and the 4x GELU MLP are
written explicitly. Inputs require `1 <= T <= max-t` and the configured width.
No ready-made ResNet or attention implementation is used.

## Run from the repository root

```sh
nix develop --command racket examples/layer-comparison/run.rkt
nix develop --command python3 examples/layer-comparison/models.py
nix develop --command dune exec --root examples/layer-comparison ./run.exe
```

Each demo runs forward, backward, an Adam step, and evaluation. The flake
supplies Jane Street Torch v0.17.0 with libtorch 2.1.2 for OCaml, independently
of the Racket and Python native runtimes.

For numerical comparison, run these in order:

```sh
nix develop --command racket examples/layer-comparison/check.rkt
nix develop --command python3 examples/layer-comparison/check.py
nix develop --command dune exec --root examples/layer-comparison ./check.exe -- "$PWD/examples/layer-comparison/ocaml-reference.txt"
```

Racket exports weights and inputs. Python and OCaml load those weights by name
before comparing outputs, so this does not rely on matching RNG streams or
initializers. The checks also exercise parameter counts, gradients, Adam,
causal isolation, and buffer registration. Generated reference data is ignored
by Git. Validation is on CPU; these examples do not establish accelerator parity.
