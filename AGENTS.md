# AGENTS.md

## Project Overview

`torch` provides Racket bindings to **libtorch** (the C++ core of PyTorch).
**v1** is in: the v0 scaffold (pipeline + handle/finalizer/last-error
substrate, `plans/v0-scaffold.md`) plus the curated tensor-op tranche,
autograd, and the `define-layer` nn system — all validated against PyTorch.
Design rationale and the closed nn-architecture decision:
`docs/design/v1-codegen-nn.md`.

The Racket package is the `torch` collection:

- `(require torch)` — the high-level API. A `tensor` is a wrapper struct whose
  native handle is reclaimed by Racket's GC; user code never frees it.
- `(require torch/nn)` — the nn layer (mirrors `import torch.nn`):
  `define-layer`, `gen:layer`, `Parameter`, `Linear`, `sgd`, `mse-loss`, initializers.
  **Naming convention:** nn layer *constructors* are PascalCase (`Linear`,
  `Conv2d`, `ConvTranspose2d`, `MaxPool2d`, `Flatten`, `Dropout`,
  `Sequential`, `Embedding`, `LayerNorm`, `GroupNorm`, `BatchNorm2d`,
  `BatchNorm1d`, `LSTM`, `GRU`), mirroring the
  `torch.nn.*` classes; their *predicates* are lowercase (`linear?`,
  `conv2d?`, `conv-transpose2d?`, `max-pool2d?`, `flatten?`, `dropout?`,
  `sequential?`, `embedding?`, `layer-norm?`, `group-norm?`,
  `batch-norm2d?`, `batch-norm1d?`, `lstm?`, `gru?`), per Racket idiom
  (`list?`, `hash?`). The functional ops keep lowercase names on `torch`
  (`conv2d`, `max-pool2d`, `flatten`, like `torch.conv2d`). The PascalCase
  constructors vs lowercase functional ops are what let `(require torch
  torch/nn)` coexist without collision (#11).
- `(require torch/foreign)` — the contracted low-level layer. Same surface,
  applied contracts; this file is the authoritative description of the API.
- `(require (submod torch/foreign unsafe))` — adds `tensor-free!` for
  deterministic release. Idempotent: a second free raises `exn:fail:contract`
  at the contract boundary instead of double-freeing.
- `(require torch/foreign/raw/*)` — the direct C FFI layer.

The native bridge is a C++ shared library, `libtorchrkt`, built with CMake and
linked against libtorch via `find_package(Torch)`.

### v1 surface

CPU-first; float32 + inferred int64 (#44) + uint8 from bytes (#58) +
float64 + the half pair float16/bfloat16 (#152: every constructor and `to`
take them, values read back through float32, `tensor->bytes` /
`bytes->tensor` carry the element bytes as they are, and `with-autocast`
runs a forward under `at::autocast` per device type, bfloat16 by default,
with `backward!` outside the form as PyTorch recommends). From
`torch`:

- v0 core: `torch-version manual-seed! randn tensor-shape tensor-numel
  tensor->list tensor->vector tensor->repr tensor->string` (+
  `tensor-dtype`; `tensor` infers int64 for all-integer data, #44, and
  builds a uint8 tensor from a byte string with no boxed conversion, #58,
  the ingestion path for image payloads and raw file buffers; the
  unprefixed property names `shape`/`dtype`/`numel` alias the tensor-
  forms; `device` doubles as query-or-construct — the
  torch.device-vs-x.device hybrid; comparison masks are first-class
  `'bool` — the dtype queries answer it and `to-dtype` casts to it)
- device: `device? device-type device-index cpu-device cuda-device
  mps-device cuda-available? cuda-if-available cuda-device-count
  mps-available? mps-if-available accelerator-if-available
  set-default-device! default-device
  with-default-device to to-device to-dtype tensor-device` — `'mps` is
  accepted wherever `'cuda` is (#13); CPU and MPS are single devices
  (index 0); `to` is PyTorch's `.to` (device, dtype, or both; identity
  when nothing changes; layers move in place through `prop:to`, which
  `gen:layer` derives, #54), and `to!` in the `unsafe` submodule is the
  in-place tensor primitive behind it; `device/c` and `dtype/c` are
  exported for user contracts
- memory: `native-memory-use` (per-device outstanding native bytes from
  the #37 ledger), `cuda-memory-stats` / `cuda-empty-cache!` /
  `mps-empty-cache!` (the caching allocators' own gauges + release, #51),
  `reclaim-native-memory!` (collect -> finalizer drain -> cache release,
  the phase-boundary release-now sequence), `finalizer-failures`
  (guarded-swallow counter) and `finalizer-diagnostics` (that counter plus
  runs, captured failure messages, and live ledger entries; also dumped at
  exit under `RKTORCH_MEM_TRACE`), `tensor-free!` (explicit synchronous
  release)
- creation: `zeros ones full fill-value/c arange eye tensor rand randn` (+ in-place
  `uniform!`); every constructor takes `#:device` / `#:dtype` chosen at
  native construction (never construct-then-move), with one exception:
  `tensor` asked for `'float16` / `'bfloat16` from a list or vector, which
  no host vector type carries, is built and narrowed on the CPU and moved
  once, so an accelerator never holds the wide copy; `#:requires-grad?`
  applied after it (integer dtypes refuse it as torch does); the shape
  constructors take dims as rest args or one list; `zeros-like` /
  `ones-like` / `full-like` / `randn-like` / `rand-like` inherit the
  reference's shape, device, dtype unless overridden (#56); `arange` stays
  float32 by default (its int64 inference is the open remainder of #56);
  `make-generator` / `generator?` / `randperm` / `draw-seed` — a CPU
  `torch.Generator` with its own stream, the permutation and the int64
  seed word drawn from it (or the global stream), for loaders (#87)
- iteration (`torch/foreign/sequences.rkt`, #199): `in-tensor` (the
  slices along the first dimension as views, Python's `for row in t`) and
  `in-flattened-tensor` (the elements as Racket numbers, row-major, copied
  to the host once; Python's `for x in t.flatten()` yields 0-d tensors)
- `length` (`torch/foreign/sized.rkt`): Python's `len` as `gen:sized`,
  shadowing racket/base's like `+`; fast defaults for lists, vectors,
  strings, hashes; a tensor's first dimension; datasets and loaders
- data (`torch/data/loader.rkt`, #87): `define-dataset` (fields, `#:init`,
  `#:length`, `#:ref`, optional `#:batch`/`#:device`, `#:contract` export,
  the `Dataset` subclass shape) over `gen:dataset` (`dataset-length`
  `dataset-ref` `dataset-batch` `dataset-device`), `tensor-dataset`
  (batches are `narrow` views or one `index-select`, device resident),
  `default-collate`,
  `dataloader #:batch-size #:shuffle? #:drop-last? #:collate #:generator`,
  `in-dataloader` (one traversal = one epoch, the generator's stream
  continuing), `in-epochs`; synchronous, single-threaded like
  `num_workers=0`; a seeded loader replays `DataLoader(generator=g)`'s
  batch order
- translation (`torch/data/translation.rkt`, #153): the PyTorch seq2seq
  tutorial's eng-fra pairs, `load-translation-pairs` (zip cached, 11445
  pairs after the tutorial's normalisation and filter) and
  `load-translation-fixture` (287 committed pairs),
  `translation-archive?`, `parse-pairs`
  `normalize-sentence`, word vocabularies with `<pad>` 0 / `<sos>` 1 /
  `<eos>` 2 (`pairs->vocabs` `encode-sentence` `decode-tokens`), and
  `pairs->tensors` padding to a width
- diffusion (`torch/vision/diffusion.rkt`, #84): `linear-schedule`
  `cosine-schedule` (betas, alphas, alpha-bars as device tensors), `q-sample`
  (closed-form `q(x_t | x_0)`), `sinusoidal-embedding`, and the layers
  `TimeEmbedding` `ResBlock` (with `#:dropout`) `AttentionBlock`
  `Downsample` `Upsample` `UNet` (the DDPM CIFAR-10 network by default:
  `#:base #:mults #:blocks #:attention #:dropout`, `#:classes` for a
  class-conditional net with a null label); the training loop is
  `examples/racket/08-diffusion.rkt`
- vision (`torch/vision/cifar10.rkt`, #84): `load-cifar10` (binary archive
  cached and unpacked in memory, float32 `[N 3 32 32]` in `[-1, 1]` plus
  int64 labels), `cifar10-dataset #:device`, `cifar10-label-names`,
  `load-cifar10-fixture` (256 committed records), `cifar10-records->tensors`,
  `tar-entries`
- ImageNet networks (`torch/vision/resnet.rkt`, #199): `resnet18`
  `resnet34` `resnet50` (`#:pretrained?`, `#:classes`; another head size
  keeps the loaded backbone and starts a fresh `fc`), `ImageNetResNet
  blocks #:block #:classes`, `Bottleneck`, `torchvision-key` (the load
  rename: `downsample` is `shortcut`, `_` is `-`); `imagenet-classes`
  (`torch/vision/imagenet.rkt`, label order) and `imagenet-preprocess`;
  the predict example is `examples/racket/14-imagenet.rkt`, parity against
  torchvision in `imagenet-parity-test.rkt` (default shell, cached weights)
- resnet (`torch/vision/resnet.rkt`, #152): `BasicBlock` and `ResNet`
  (`#:classes #:base #:blocks`, ResNet-18 for 32x32 images by default,
  bias-free convolutions under `BatchNorm2d`); the training loop with
  device-side augmentation, SGD under `one-cycle-lr` and `with-autocast` is
  `examples/racket/09-resnet.rkt`
- transforms (`torch/vision/transforms.rkt`, #152): `random-horizontal-flip
  #:p` and `random-crop #:padding` on an image batch where it lives; each
  takes `#:generator` and draws one seed per batch from it, so a seeded
  loader replays its augmentation (the draws are the transform's own, not
  torchvision's). The pretrained preprocessing (#199) on an image or a
  batch: `convert-image-dtype` (uint8 0-255 to float 0-1 and back, the
  torchvision scaling), `resize #:antialias?` (short side or `(h w)`,
  float only; two matmuls with host-computed bilinear or Pillow-filter
  weights, since `upsample_bilinear2d` needs the unmarshalled `float?`),
  `center-crop` (offsets round half to even), `normalize`,
  `imagenet-normalize`, `imagenet-mean`, `imagenet-std`
- pretrained weights (`torch/vision/weights.rkt`, #199):
  `pretrained-weights` fetches a torchvision ImageNet checkpoint, exported
  by `scripts/export-weights.py`, from the `weights-v1` release into the
  cache (`RKTORCH_WEIGHTS_DIR`, `RKTORCH_WEIGHTS_URL`), checking the size
  and SHA-256 recorded in the module before the rename into place;
  `pretrained-weights-cached?`, `pretrained-weights-names`. The files keep
  torchvision's key names; `load-state! #:rename` maps them
- image reading (`torch/vision/image.rkt`, #199): `decode-image` (bytes)
  and `read-image` (a path) to a uint8 `[C H W]` tensor, `#:mode
  'unchanged 'gray 'gray-alpha 'rgb 'rgba`, `#:device`; JPEG and PNG
  through `stb_image` behind `tr_image_decode`, the header from nixpkgs'
  `stb` found by pkg-config like libsndfile and compiled once in
  `src/torchrkt/detail/stb_image.c`; PNGs match
  torchvision.io exactly, JPEGs within a count or two
  (`image-parity-test.rkt`, torchvision only in the default shell)
- generative examples on MNIST (#152): `examples/racket/10-dcgan.rkt` (a
  DCGAN shrunk to 28x28, `ConvTranspose2d` and `BatchNorm2d` in the
  generator, `leaky-relu` in the discriminator, two `adam`s at 2e-4 with
  `#:beta1 0.5`) and `11-vae.rkt` (the linear VAE, the reparameterization
  with the caller's noise, the reference loss over the batch); both write a
  10x10 sample grid per epoch through `image-grid` and `write-ppm`
- images (`torch/vision/ppm.rkt`, #155): `image-grid #:columns #:padding
  #:pad-value` (torchvision's `make_grid` layout, on the device) and
  `write-ppm #:range` (binary P6, `save_image`'s quantization; a uint8 image
  as it is)
- shape: `reshape view transpose permute squeeze unsqueeze cat stack flip`
- elementwise: `add sub mul div pow neg exp log sqrt relu sigmoid tanh silu
  leaky-relu clamp`
  (binary ops take a real on either side)
- operators: `+ - * /` shadow racket/base rkt-polars-style (numeric fast
  path to racket/base, tensor operands dispatch to add/sub/mul/div, chains
  fold left); `@` is matmul, like Python's `a @ b`; `t`/`Σ` are terse
  aliases for transpose/sum; unary `T` reverses all axes like Python's `x.T`
  (use `transpose` on the last two axes for batched attention);
  `~> ~>> lambda~> lambda~>>` are re-provided
  from the `threading` library (dep `threading-lib`, prefetched offline by
  the `racket-deps` fixed-output derivation in flake.nix)
- reductions: `sum mean max min argmax softmax log-softmax`
- ordering and sampling (`torch/foreign/order-ops.rkt`, tranche 7, #153):
  `topk` and `sort` answer `(values entries indices)`; `argsort`;
  `multinomial #:replacement? #:generator` (no-retry RNG wrap; seeded CPU
  draws match PyTorch's); `sort` on a non-tensor is racket/base's
- linalg: `matmul mm mv dot`; out: `item to-dtype`
- autograd: `requires-grad! requires-grad? backward! grad has-grad?
  maybe-grad detach with-no-grad grad-enabled?`; in-place
  `sub! zero! mul! copy! addcmul! addcdiv! lerp! zero-grad!`
- autocast: `with-autocast call-with-autocast autocast-enabled?
  autocast-dtype` (`torch/foreign/autocast.rkt` over
  `cpp/src/torchrkt/autocast.cpp`; per thread and per device type like
  grad mode, restored by `dynamic-wind`, the cast cache dropped on exit)

**Name shadowing convention:** ops colliding with racket/base or racket/list
(`exp log sqrt tanh max min argmax sort`) are generic — tensors hit libtorch,
anything else defers to the original — so `(require torch)` never breaks
numeric code. New ops that collide must follow the same dispatch pattern
(check racket/base first — `tanh` was missed initially and broke numeric
callers), and scribble examples need
`(for-label (except-in racket/base abs cos exp log sin sort sqrt max min
length + - * /) torch)` — the `except-in` alone leaves those names
unbound, so `torch` has to follow it. Each chapter carries that line
itself: re-exporting it from a shared module tags every link to that
module rather than to `torch`, so the links miss their entries.
Dispatching named ops carry dependent (`->i`) contracts so the wrong shape
gets contract blame, not a runtime error; the `+ - * / @` operators are
provided as plain renames (no contract overhead on the numeric fast path),
per `foreign/operators.rkt`.

From `torch/nn`: `define-layer procedure->Layer gen:layer layer? Parameter Buffer LayerList LayerHash parameters
named-parameters buffers children forward Linear Conv2d MaxPool2d Flatten Dropout
Sequential Embedding LayerNorm ConvTranspose2d GroupNorm BatchNorm2d BatchNorm1d
LSTM GRU sgd adam rmsprop step! zero-grads! clip-grad-norm! learning-rate
set-learning-rate! step-lr multi-step-lr exponential-lr cosine-annealing-lr
linear-lr one-cycle-lr lambda-lr ema ema-update! ema-average cross-entropy
nll-loss mse-loss binary-cross-entropy-with-logits huber-loss l1-loss
kaiming-uniform uniform-init normal-init fan-in`. The functional
transformer primitives (`gelu tril triu masked-fill embedding layer-norm`,
tranche 3, #22), the UNet ones (`conv-transpose2d group-norm silu
clamp`, tranche 4, #84; `upsample-nearest2d` over `repeat_interleave`, tranche
5) and the classic vision ones (`batch-norm leaky-relu flip linear`, tranche 6,
#152; `Linear` runs on the fused `linear`, so autocast casts the whole affine
map; `BatchNorm2d`/`BatchNorm1d` keep `running-mean`, `running-var` and
`num-batches-tracked` as `Buffer`s that ATen updates in place in `'train`
mode) live on `torch` beside the other functional ops; the GPT
causal-mask idiom is `(masked-fill scores (eq (tril (ones T T)) 0) -inf.0)`. `define-layer` is the Python-style
`nn.Module` analog: `#:init` is the constructor body and assigns declared
fields with `set!`, a field's value classifies it at construction
(`Parameter?`, `Buffer?`, `layer?`, `#f` for absent, anything else plain),
`(with-mode body)` binds `mode` (`'train` or `'eval`, predicates `training?`/`evaluating?`) in `#:forward` to the instance's own mode (`train!`/`eval!` set it and recurse; every layer starts in `'train`),
models are plain struct trees owned by the GC (no global parameter store),
and `prop:procedure` makes `(net x)` work like `__call__`. `LSTM` and `GRU` (`nn/recurrent.rkt`, #153) are
`define-layer` forms whose parameter set depends on `#:num-layers` and
`#:bidirectional?`: `parameters-by-key` registers them under PyTorch's own
names (`weight_ih_l0` .. `bias_hh_l1_reverse`) the way `children-by-key`
registers children, and a rest-argument `#:forward` lets an initial state
follow the input. Applying one answers `(values output h-n [c-n])`. On CUDA
the weights are flattened for cudnn (`cudnn-rnn-flatten-weight`, in place,
the parameters keep their identity) whenever their device or dtype differs
from the placement the last flattening was built for, so `to`'s identity
case costs nothing and a transient OOM is retried rather than latched. `clip-grad-norm!` (`nn/clip.rkt`) keeps its scale on the
device. Layer init mirrors
PyTorch RNG consumption (`nn.Linear.reset_parameters`), so a shared
`manual-seed!` yields bit-comparable parameters — the MLP cross-test relies
on this.

**REPL parity.** A tensor prints in the Racket REPL exactly as it does in the
Python REPL — `tensor([[ 1.5410, -0.2934], [-2.1788, 0.5684]])` — via
`prop:custom-write`, reproduced from the data + shape in `foreign/format.rkt`
(PyTorch's `tensor(...)` framing isn't in libtorch's C++ printer). Two
accessors expose the two forms explicitly: `tensor->repr` is the PyTorch repr
(what the REPL shows); `tensor->string` is ATen's C++ `operator<<` text. The
repr reproducer covers the common case (CPU float32, fixed-point, precision 4)
and falls back to the ATen form for scientific-notation values; see the TODO in
`foreign/format.rkt`.

Deferred (see plan): `native_functions.yaml` codegen, `nn.Module` macros, the
broader ATen surface, and the portable raco-catalog candidate story.

## The libtorch source knob

`flake.nix` has `torchSource = "bin" | "python"`:

- **`bin`** (default) — `pkgs.libtorch-bin`. Small prebuilt download, fast cached
  CI on `aarch64-darwin` + `x86_64-linux`. Parity with Python torch is *tolerant*
  (the cross-test uses a float tolerance), because the C++ side may be a
  different patch version than the Python torch.
- **`python`** — `pkgs.python3Packages.torch`. Builds against the *same* libtorch
  the parity script imports, so seeded `randn` is **bit-exact** — at the cost of
  a heavy (often uncached on darwin) from-source build.

## Build Commands

[`docs/building.md`](docs/building.md) is the same guide written for people;
a change to a build target or a shell belongs in both. The ordered local loop
for a change, and the gates it has to pass before it is done, are the
`cpp-dev` and `racket-dev` skills under `.claude/skills/`.

```bash
nix build              # builds cpp, installs the pkg, runs raco test + examples
nix build .#cpp        # CMake build + gtest only
nix flake check        # the Nix checks: cpp, format, tidy, line-count, racket
                       # (CI also runs the Resyntax gate and codegen-drift)
nix develop            # dev shell (includes a Python with `torch`)
nix develop .#ci       # lean shell without Python torch (used by the lint job)
nix develop .#cuda     # linux-only: CUDA-linked shim + host driver (#14)
nix develop .#ocaml    # adds OCaml + Jane Street's Torch bindings (reference)
./result/bin/torch  # runs (module+ main): prints version + a 2x2 draw
```

Inside `nix develop`:

```bash
cmake -S cpp -B cpp/build -G Ninja -DBUILD_TESTING=ON
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure

raco test torch/          # FFI unit tests (+ self-skipping parity test)
raco test examples/test/     # literate-example runners
racket -ie "(require torch)"   # REPL with the package loaded
                               # (`racket -l torch` runs module+ main instead)

racket scripts/coverage.rkt --changed   # expression coverage (#173); exits
                               # non-zero below the floor and lists the lines
                               # of the files you touched that no test reaches

resyntax analyze --local-git-repository . origin/master   # lint gate
                                     # (CI fails on any suggestion; scans
                                     #  changed files, torch/ included)
resyntax fix --directory torch
raco review torch/**/*.rkt
```

`raco review` does not expand macros, so the pure re-export facades
(`main.rkt`, `foreign.rkt`) and `info.rkt` carry a `#|review: ignore|#` directive.

### PyTorch parity

The default `nix develop` shell ships a Python with `torch`, so you can explore
PyTorch behaviour beside the Racket bindings and run the real cross-test:

```bash
nix develop --command python3 -c 'import torch; print(torch.__version__)'
nix develop --command raco test torch/tests/python-cross-test.rkt
nix develop --command raco test torch/tests/generated-parity-test.rkt
```

Where python3 can't `import torch` (the sandboxed `nix build`, or the lean
`.#ci` shell), the tests self-skip, so `nix build` / `raco test` stay green.

### Accelerators

`accelerator-if-available` picks CUDA, then MPS, then CPU — what the examples'
`pick-device` returns, and the analogue of PyTorch's
`torch.accelerator.current_accelerator()`. The device suite guards its CUDA and
MPS bodies separately; the example runners train on whichever accelerator the
helper returns. Both verify real hardware where it exists and self-skip where it
doesn't — never assume a case ran because it was green.

**Darwin needs no special shell.** The `aarch64-darwin` `libtorch-bin` ships the
Metal backend, so plain `nix develop` is the MPS verification environment —
there is no `.#mps` counterpart to `.#cuda` (which exists only because CUDA
needs a differently-linked libtorch plus the host driver). Confirm with
`nix develop --command racket -e '(require torch)(mps-available?)'`.

**The one MPS kernel gap.** libtorch 2.9 registers `aten::_ctc_loss` for CPU
and CUDA only. `ctc-loss` (`torch/nn/loss.rkt`) therefore marginalizes MPS
frames on the CPU and moves the scalar back; `to-device` is differentiable, so
the gradient returns to the MPS graph and the rest of a model — the 07-asr
encoder, attention decoder, and `adam` — stays on the GPU. Every other op the
speech arc uses has an MPS kernel, so `pick-device` must keep returning
`accelerator-if-available` unmodified: routing darwin to the CPU to dodge this
one op is what the carve-out exists to avoid. The second gap is
`aten::native_group_norm_backward`: the `GroupNorm` layer
(`torch/nn/group-norm.rkt`) normalises on the CPU under MPS the same way, so
the diffusion UNet trains on the GPU there with its norms round-tripped.

## Architecture

### C++ (`cpp/`)

- `include/torchrkt/c_api/*.h` — the `extern "C"` FFI surface (global, random,
  tensor, creation, shape_ops, elementwise, reduce, linalg, autograd).
  Integer-status + size-then-fill + `tr_last_error` contract; opaque
  `tr_tensor` handles are returned by constructors/ops and freed by
  `tr_tensor_free`.
- `src/torchrkt/*.cpp` — translation layer; catches C++ exceptions, returns
  status codes / NULL. `detail/tensor_handle.hpp` (in `src/`, private)
  completes the opaque struct over a `torch::Tensor`
  (`generator_handle.hpp` does the same for `tr_generator`);
  `detail/op_call.hpp` holds the boundary helpers (`alloc_result` and
  `null_arg` for tensor-returning ops, `alloc_handle<H>` for any other
  opaque handle; `status_call` and `null_arg_status` for the int-status
  in-place shape; `alloc_results` for an op with several Tensor returns,
  an int status plus one `tr_tensor**` out pointer per return, every one
  NULL unless the whole call succeeded) every op body reduces to — new ops must use them rather
  than hand-rolling try/catch.
- `tests/torchrkt/{random,ops,autograd,generated_golden,generated_tranche2..7}_test.cpp`
  — GoogleTest goldens per family (generated families get a C-boundary
  golden: a correctness case + a null/length-guard case).
  `c_api_compile_test.c` proves the headers are valid C (add a
  function-pointer line for at least one representative of each new op
  family, plus any function whose signature shape is new).

### Racket (`torch/`)

Thin re-export facades over small modules (target ≤ 500 lines/file).

**Import convention:** library modules use `(require (only-in ...))` with
explicit, alphabetized name lists — never whole-module requires — so each
file documents exactly what it pulls in. Exemptions, each marked with a
comment at the require site: pure re-export facades (`main.rkt`,
`foreign.rkt`, `nn.rkt`), and macro-heavy modules whose expansions need the
module's full export set (`racket/runtime-path`, `syntax/parse/pre`).

- `info.rkt` — package metadata + native-library pre-install hook.
- `main.rkt` — high-level facade (re-exports `foreign.rkt`).
- `foreign.rkt` — the contracted layer + the `unsafe` submodule.
- `data/dataset.rkt` — `define-dataset` and `gen:dataset`;
  `private/definer.rkt` — the clause grammar it shares with `define-layer`.
- `vision/cifar10.rkt` — CIFAR-10 loader and dataset, `vision/fixtures/`
  its 256-record fixture and `vision/fixtures/images/` the reader's
  synthetic JPEGs and PNGs (`scripts/gen-image-fixtures.py`);
  `vision/image.rkt` — the image reader; `vision/diffusion.rkt` — DDPM schedules, `q-sample`
  and the UNet layers.
- `data/loader.rkt` — `tensor-dataset`, `dataloader`, `in-dataloader`,
  `in-epochs`, re-exporting `data/dataset.rkt`; `data/mnist.rkt`,
  `data/text.rkt`, `data/translation.rkt` — the modality loaders (moving
  under #88).
- `foreign/ops.rkt` — version/seed + marshalling (`item`, `to-dtype`,
  `uniform!`, `to`); `foreign/creation-ops.rkt` — the constructors
  (`zeros` .. `rand`, `tensor`, `arange`, `eye`, the `*-like` family, with
  placement and `#:requires-grad?` handled once); `foreign/tensor-ops.rkt`
  — the op tranche (and the
  shadow-dispatch convention); `foreign/order-ops.rkt` — `topk` `sort`
  `argsort` `multinomial`; `foreign/autograd-ops.rkt` — autograd +
  `with-no-grad` + in-place ops; `foreign/structs.rkt` — the `tensor`
  wrapper (`prop:cpointer`, shape cached at wrap time, allocator/deallocator
  finalizer); `foreign/error.rkt` — `check-ok` / `check-handle`;
  `foreign/format.rkt` — the PyTorch-repr reproducer.
- `foreign/raw/*.rkt` — direct FFI, one module per C translation unit:
  `syntax` (the pure FFI definer + `_Tensor` cpointer), `pressure` (no
  FFI of its own: the collection policy under the ledger, the two troughs
  and the capacity backstop, #145), `memory` (the lifetime substrate:
  frees, pressure ledger, `tensor-allocator`, op-definer macros),
  `global`, `tensor`, `random`, `creation`,
  `shape-ops`, `elementwise`, `reduce`, `linalg`, `autograd`.
  **`docs/internals.md` is the canonical memory-management narrative**
  (lifetime chain, phantom-bytes pressure, typed OOM + retry). The
  rules agents must not break: every tensor-returning binding carries
  `tensor-allocator` — or `tensor-allocator/rng` for bindings that draw
  from the global RNG stream (randn/rand; ops flagged `rng` in the
  codegen allowlist) so a retry can never double-draw and break seeded
  parity; a binding with several tensor outputs carries
  `tensor-allocator/outputs` (or `/outputs/rng`), which registers and
  accounts every handle inside one atomic section; never a bare
  `(allocator ...)` wrap (skips the ledger).
  Explicit synchronous release goes through the raising,
  finalizer-cancelling `tr-tensor-free/checked`; OOM reaches users as
  `exn:fail:rktorch:oom` (catch by type, not message).
- `nn.rkt` — pure re-export facade over `nn/` (`layer.rkt` = `gen:layer`, `LayerList` +
  the `define-layer` macro, whose `#:forward` takes a rest argument, whose
  fields admit `parameters-by-key` beside `children-by-key`, and whose
  `#:on-move` body runs after a `to` that rebound anything; `parameter.rkt`, `buffer.rkt`, `linear.rkt`,
  `init.rkt`, `optim.rkt`, `ema.rkt`, `loss.rkt`, `recurrent.rkt`,
  `clip.rkt`).
- `private/install-torchrkt-native.rkt` — stages `libtorchrkt.*` into
  `native-libs/` from `TORCHRKT_NATIVE_LIB_PATH` (set by the Nix build/shell).
  Every staging path (here and the flake's three shell ones) writes a temp file
  and `rename(2)`s it into place, so restaging under a live process is safe:
  rename leaves the old inode alive and anything already running keeps
  executing the old lib. **Restart the REPL to pick up a new shim — it does not
  hot-swap.** Never reintroduce an in-place `cp`; it does not merely fail to
  swap, it corrupts the running process (#72). Shell entry stages only when the
  staged bytes differ from the shim that shell wants (`cmp`), so `.#cuda` no
  longer restages on every entry, returning to the default shell restores the
  CPU shim, and a deleted or truncated shim is replaced rather than skipped. After
  changing C++, `nix run .#copy-native-libs` (or re-enter the shell) before
  `raco test`.

### Codegen (`codegen/`)

The ATen generator (v2/A, #2): `nix run .#codegen` (equivalently
`nix develop --command python3 -m codegen`, but with a much smaller
closure) reads `codegen/allowlist.txt` against the **vendored** schema in
`codegen/aten/` (pinned to the C++ libtorch 2.9.0 — see the README there;
never the dev-shell python torch's copy) and emits, with DO-NOT-EDIT
headers:

- `cpp/{include/torchrkt/c_api,src/torchrkt}/generated/<shard>.{h,cpp}` —
  bodies reduce to the `op_call.hpp` helpers; clang-format is run by the
  generator; `generated/sources.cmake` is included from `cpp/CMakeLists.txt`
- `torch/generated.rkt` — the UNSTABLE uncontracted surface: one compact
  `define-generated-op` form per allowlist entry. The hand-written macro
  in `torch/foreign/define-generated.rkt` owns the expansion into raw FFI
  binding + wrapper, so Racket marshalling knowledge lives in Racket, not
  in Python string templates. Promotion into `torch/foreign.rkt` is
  hand-curated.
- `torch/tests/generated-parity.rktd` — manifest driving the generated-op
  battery in `generated-parity-test.rkt`; every new allowlist line needs an
  input recipe in that test (`'device-only` for an op with no CPU kernel,
  which then needs a device-guarded test of its own)

Conventions:

- **Extend the allowlist instead of hand-writing** when an op fits the IR
  (Tensor / Scalar→double / int64 / bool / IntArrayRef / TensorList args,
  one or more Tensor returns). Unsupported signatures are skipped with a report —
  widening the IR is a generator change, not a hand-written shim.
- Optional *types* are in the IR: `Tensor?` is a NULL pointer, `int?` and
  `Scalar?` carry a presence flag, `int[]?` a length plus flag, and
  `ScalarType?` a -1 sentinel, and `Generator?` a `tr_generator` handle
  (NULL for the global stream; an op that draws still needs the allowlist
  `rng` flag). In-place ops (`add_`) emit a mutable receiver
  plus an integer status. An op with several Tensor returns (`topk`,
  `sort`, #154) emits an integer status plus trailing out pointers in C
  and a `#:returns N` clause in Racket, where it answers multiple values
  in schema order; any non-Tensor return still skips. A private ATen name
  loses its leading underscore on the Racket side
  (`_cudnn_rnn_flatten_weight` is `cudnn-rnn-flatten-weight`). Schema *defaults* (`int dim=0`) are still
  flattened to required arguments on the unstable surface — defaults are a
  curated-facade concern.
- Generated output is committed (AOT); CI's `codegen-drift` job regenerates
  and fails on any diff, so never edit generated files by hand.
- `generated/` is exempt from the C++ 500-line gate (shard size is the
  generator's concern).
- The golden-equivalence proof lives in
  `cpp/tests/torchrkt/generated_golden_test.cpp`: the generated linalg four
  (`tr_gen_{matmul,mm,mv,dot}`) stay permanently allowlisted and bit-checked
  against the authoritative hand-written family.

## Comment policy

Before submitting a PR, actively REMOVE comments — delete, don't
shorten. Code should be self-documenting; when tempted to keep a
comment, first try to encode it in a name or structure (rename, extract
a helper, tighten a contract — renaming a function to make its behavior
explicit beats any comment) and then delete it. A correct explanation of
what the code visibly does is still noise. Contract narration on a
declaration whose name and signature carry the meaning is noise.

Documentation of a form or function — what it does and how to use it —
belongs in `torch/scribblings/*.scrbl`, not in a comment above the
definition.

The rare comments that stay carry something no name or structure could
express, in one or two lines:

- a genuine invariant invisible in the code (finalizers run in atomic
  mode; the OOM retry must compose OUTSIDE the allocator wrap)
- a strange cross-boundary interaction (FFI marshalling gotchas, noexcept
  walls, cuBLAS's missing int64 matmul)
- a documented deviation from reference behavior (e.g. a PyTorch-parity
  gap that is deliberate)

Never keepers: narration of what the next line does, why a change was
made (commit message's job), review-round archaeology, restated names, or
issue-number tags on every mention of a mechanism. Exempt: the literate
examples (`examples/racket/*.rkt`, prose-by-design) and the python
parity twins (`examples/python/*.py`, whose docstrings identify the
Racket twin they mirror).

## Validating arguments

Exported definitions carry their contract at the definition site with
`define/contract-out` (`torch/private/contract.rkt`), which expands to a
`define` plus `(provide (contract-out ...))`.  One form, so the contract
sits where a reader already is and the name is not repeated in a separate
`provide` block.  Not `define/contract`: it blames the defining module
rather than the caller, which is backwards for a library reporting what a
user passed.

Inside `torch/foreign/` a name the library itself calls uses
`define/checked-out` instead.  That exports the definition plainly and puts
the contracted name in a `checked` submodule, so a sibling requiring the
module reaches the definition while `foreign.rkt` requires
`(submod ... checked)` and re-exports it.  Without the split, contracting
`add` would contract `operators.rkt`'s `t+`, and contracting `tensor?`
would make every contract in `contracts.rkt` cross a second boundary.

Which form to use is decidable, not a judgement call: **does a module under
`torch/foreign/` import this name?**  If yes `define/checked-out`, if no
`define/contract-out`.  A test in `torch/tests/contract-out-test.rkt`
asserts no module under `torch/foreign/` requires a `checked` submodule, so
the fast path cannot be recontracted by a later edit.  The same rule and
the same test cover `torch/nn/`, with `nn.rkt` as the facade.

A layer built with `define-layer` carries its constructor contract in a
`#:contract` clause.  The clause is the export: it provides the constructor
under that contract and the predicate, under its lowercase name
(`Conv2d`/`conv2d?`, `MaxPool2d`/`max-pool2d?`; `#:predicate` overrides the
derived name), so a layer file has no `provide` block and no `rename-out`.
`->i` states a cross-argument invariant that used to be an `unless` guard.
A forward formal written `[x : image-batch/c]` states what the layer accepts
there, which is where a rank or shape check belongs; the violation names the
layer and the contract, and the check is built once at the definition.

Two layers carry no contracts: `torch/generated.rkt` (codegen output, the
unstable surface) and `torch/foreign/raw/` (the FFI bindings, where the
contract arrow would shadow the one `_fun` matches).  A name promoted
from either is contracted at the promotion site with the value form,
`(define/contract-out narrow (-> ...) g:narrow)`.  A struct predicate has
no definition to wrap; its `checked` provide is written by hand beside
the `struct`.  `foreign.rkt` is a pure re-export module: it requires each
module's `checked` submodule, excludes the plain names those replace, and
provides the result.

`unless`+`error` stays for what a contract cannot see -- checks against
parsed content (WAV chunk structure), filesystem state (symlink
containment), or values computed mid-body.  A guard that only inspects an
argument's shape belongs in the signature.

Name non-obvious contracts with `flat-named-contract` so the violation
message still says `sample-rate` rather than an inlined `and/c` chain.

`raco review` lints unexpanded, so it cannot see the generated `provide`
and reports an exported definition used nowhere else in its own module as
"identifier is never used".  Mark the ones it flags `;; noqa`, the same
per-identifier directive the re-export shims already use;
`#|review: ignore|#` suppresses a whole file and is too blunt here.  It
flags fewer than you would expect: a definition that passes its own name
as a quoted symbol, as `(check-ok rc 'audio-info)` does, already counts as
used.  (`resyntax`, the CI gate, is unaffected.)

### What the strategies cost

`scripts/bench-contract-overhead.rkt`, lab host, rounded.  The tensor
columns are noisy: the 8x8 `add` baseline moves ~20% between runs, enough
that its 1.0x-1.1x rows should be read as "lost in the noise" rather than
as a measured overhead.  Every callee is imported from its defining
module, never from the `torch` facade, or the baseline would already be
paying a crossing.

| | flat `->` | `->i` + `#:pre` | cached shape read | 8x8 tensor `add` |
|---|---|---|---|---|
| bare, ns/call | 10 | 12 | 7 | ~10000 |
| `unless`+`error` | 1.2x | 2.1x | 10x | 1.0x |
| `define/contract` intra | 9x | 19x | 25x | 1.1x |
| `define/contract` cross | 9x | 20x | 25x | 1.1x |
| `define/contract-out` intra | 1.0x | 1.0x | 1.0x | 1.0x |
| `define/contract-out` cross | 4x | 19x | 19x | 1.1x |

**How much the contract does matters more than what it wraps.**  A flat
`(-> tensor? tensor? tensor?)` on an allocating op reads 1.0x-1.1x, which
this benchmark cannot separate from its own noise -- so call it too small
to resolve here, not free.  The contract `add` actually ships is
resolvable, because it is measured as a paired facade-vs-raw comparison
in one run: **~1.4-1.7x** across runs, 13500 ns through `torch` against
8950 raw on one, 18500 against 11000 on another.  That contract,
`binary-arith/c`, does not inspect shapes or dtypes -- it checks the
operands are tensors or reals, makes the admissible type of `b` depend on
whether `a` is a tensor, and checks the result is a tensor.

The benchmark does not separate the `->i` machinery from the extra
predicate work it does (`or/c` dispatch, and re-testing `(tensor? a)` to
select `b`'s contract), so that ratio is the cost of the whole
contract, not of dependency as such.  Either way the carve-out providing
`+ - * / @` as plain renames is well founded, and the richest contracts
on the hot tensor surface are the expensive part of `foreign.rkt`.

**Cheap accessors are the other extreme.**  `tensor-shape` is a struct
field read at ~7 ns, so a boundary contract on it is 19x and even a hand
guard is 10x, because `tensor?` alone is ~60 ns.  There the question is
whether to validate at all.

**`define/contract` charges the same inside a module as across it**,
because it wraps the definition rather than the export.
`define/contract-out` is 1.0x internally in every workload and charges
only at a real boundary -- the second reason to prefer it, alongside
blame.

`log-mel-spectrogram` runs ~3 ms over the speech fixture, measured.  What
share of that is contract is *not* measured -- isolating it would need an
uncontracted copy of the pipeline, which has not been built.

## CI

`.github/workflows/nix.yml`: `nix flake check` on `ubuntu-latest` +
`macos-latest`, a `resyntax` lint gate, and a `codegen-drift` job
(regenerate + fail on porcelain). (A `raco-catalog` workflow is deferred
until the portable native-candidate story exists — libtorch is too large to
bundle the xgboost way.)

### Binary cache (#79)

The lab host serves its `/nix/store` as a signed binary cache (Harmonia
behind `tailscale serve`, tailnet-only) at
`https://lab.tailb6061b.ts.net`, key
`lab.tailb6061b.ts.net-1:/dlX81jfwuolbO7oiV11HoXe1ib6gMik1XPG2x8u3xc=`.
Every job installs Nix through `.github/actions/nix-cache`, which on
Linux first joins the tailnet as an ephemeral `tag:ci` node
(`tailscale/github-action`, OAuth client in the `TS_OAUTH_CLIENT_ID` /
`TS_OAUTH_SECRET` secrets) and lists the host as an extra substituter
beside `cache.nixos.org`; `fallback = true` and a 5 s connect timeout
mean a missing secret (fork PRs), a down host, or the macOS runner (no
tailnet) degrade to a normal uncached build, never a failure. The cache
serves whatever the host has built: master's C++ shim and the tidy and
format checks after a local `nix build`, the CUDA shim, the codegen
Python. A nightly `nix-cache-warm.timer` is meant to pin master's
closures under `/nix/var/nix/gcroots/cache-warm` but currently builds
nothing (#131), so a miss after a master merge means nobody has built
that commit on the host yet. CI does not push. To use the cache from
another tailnet machine, add the same two lines as `extra-substituters`
/ `extra-trusted-public-keys` to the daemon's `/etc/nix/nix.conf` (a
multi-user Nix ignores them in `~/.config/nix/nix.conf` unless the
caller is in `trusted-users`).
