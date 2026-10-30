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
  the #37 ledger), `native-memory-unaccounted` (per device, the
  allocator's bytes past the ledger's as of the last trough, charged to
  the collector as the shadow, #213), `cuda-memory-stats` / `cuda-empty-cache!` /
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
  continuing), `in-epochs`; `#:workers 0` builds batches on the calling
  thread like `num_workers=0`, `#:workers n #:prefetch p` on n parallel
  threads (#168), batch j by worker j mod n, in order, at most n*p ahead,
  each worker seeded like PyTorch's; torch calls inside a worker are
  serialized through the main thread until #296; a seeded loader replays
  `DataLoader(generator=g)`'s batch order for any worker count
- translation (`torch/text/translation.rkt`, #153): the PyTorch seq2seq
  tutorial's eng-fra pairs, `load-translation-pairs` (zip cached, 11445
  pairs after the tutorial's normalisation and filter) and
  `load-translation-fixture` (287 committed pairs),
  `translation-archive?`, `parse-pairs`
  `normalize-sentence`, word vocabularies with `<pad>` 0 / `<sos>` 1 /
  `<eos>` 2 (`pairs->vocabs` `encode-sentence` `decode-tokens`), and
  `pairs->tensors` padding to a width
- diffusion (`torch/vision/diffusion.rkt`, #84): `linear-schedule`
  `cosine-schedule` (betas, alphas, alpha-bars as device tensors), `q-sample`
  (closed-form `q(x_t | x_0)`), `sinusoidal-embedding` (`sinusoidal-positions`
  in its `'halves` layout over int64 timesteps), and the layers
  `TimeEmbedding` `ResBlock` (with `#:dropout`) `AttentionBlock` (one head
  over the H·W tokens through `scaled-dot-product-attention`, its `q` `k`
  `v` `proj` children kept as `Linear`s)
  `Downsample` `Upsample` `UNet` (the DDPM CIFAR-10 network by default:
  `#:base #:mults #:blocks #:attention #:dropout`, `#:classes` for a
  class-conditional net with a null label); the training loop is
  `torch/scribblings/examples/08-diffusion.rkt`
- vision (`torch/vision/cifar10.rkt`, #84): `load-cifar10` (binary archive
  cached and unpacked in memory, float32 `[N 3 32 32]` in `[-1, 1]` plus
  int64 labels), `cifar10-dataset #:device`, `cifar10-label-names`,
  `load-cifar10-fixture` (256 synthetic records from
  `scripts/gen-dataset-fixtures.py`), `cifar10-records->tensors`,
  `tar-entries`
- image folders (`torch/vision/image-folder.rkt`, #199): `image-folder
  #:transform #:extensions #:device` (torchvision's `ImageFolder`, items
  decoded as RGB on demand), `image-folder-classes`,
  `image-folder-samples`; `torch/vision/hymenoptera.rkt` the tutorial's
  ants and bees (`hymenoptera-dataset` `hymenoptera-root`
  `hymenoptera-cached?`, zip checked by size and SHA-256 through
  `torch/private/download.rkt`, shared with the weights); the fine-tune
  example is `torch/scribblings/examples/15-finetune.rkt`, with a
  `random-resized-crop` in the transforms
- ImageNet networks (`torch/vision/resnet.rkt`, #199): `resnet18`
  `resnet34` `resnet50` (`#:pretrained?`, `#:classes`; another head size
  keeps the loaded backbone and starts a fresh `fc`), `ImageNetResNet
  blocks #:block #:classes`, `Bottleneck`, `torchvision-key` (the load
  rename: `downsample` is `shortcut`, `_` is `-`); `imagenet-classes`
  (`torch/vision/imagenet.rkt`, label order) and `imagenet-preprocess`;
  the predict example is `torch/scribblings/examples/14-imagenet.rkt`,
  parity against torchvision in `torch/tests/imagenet-parity-test.rkt` (default shell,
  cached weights)
- VGG (`torch/vision/vgg.rkt`, #202): `vgg16-features #:pretrained?`, the
  convolutional half of VGG-16 as one `features` child, a `Sequential` of
  31 steps in torchvision's slots (so `features.N.weight` loads without a
  rename; walk it with `in-layers` for activations); `vgg16-features?`.
  The style-transfer example is
  `torch/scribblings/examples/16-style-transfer.rkt`; its runner's test
  holds it to `torch/tests/python/style_transfer.py` (default shell, cached
  weights).
- resnet (`torch/vision/resnet.rkt`, #152): `BasicBlock` and `ResNet`
  (`#:classes #:base #:blocks`, ResNet-18 for 32x32 images by default,
  bias-free convolutions under `BatchNorm2d`); the training loop with
  device-side augmentation, SGD under `one-cycle-lr` and `with-autocast` is
  `torch/scribblings/examples/09-resnet.rkt`
- transforms (`torch/vision/transforms.rkt`, #152): `random-horizontal-flip
  #:p` and `random-crop #:padding` on an image batch where it lives; each
  takes `#:generator` and draws one seed per batch from it, so a seeded
  loader replays its augmentation (the draws are the transform's own, not
  torchvision's). The pretrained preprocessing (#199) on an image or a
  batch: `convert-image-dtype` (uint8 0-255 to float 0-1 and back, the
  torchvision scaling), `resize #:antialias?` (short side or `(h w)`,
  float only; two matmuls with host-computed bilinear or Pillow-filter
  weights; `upsample_bilinear2d` is not bound),
  `center-crop` (offsets round half to even), `normalize`,
  `imagenet-normalize`, `imagenet-mean`, `imagenet-std`
- pretrained weights (`torch/vision/weights.rkt`, #199):
  `pretrained-weights` fetches a torchvision ImageNet checkpoint, timm's
  `tv_in1k` safetensors on Hugging Face pinned to a commit, into the cache
  (`RKTORCH_WEIGHTS_DIR`; `RKTORCH_WEIGHTS_URL` is a hub mirror), checking
  the size and SHA-256 recorded in the module before the rename into place
  (`scripts/check-weights.py` confirms the tensors are torchvision's);
  `pretrained-weights-cached?`, `pretrained-weights-names`. The files keep
  torchvision's key names; `load-state! #:rename` maps them. A checkpoint
  with `keys` (VGG-16's `features.`) is fetched as a byte range, its header
  and those entries (read no further, so a mirror that ignores `Range`
  still works), checked by that prefix's size and SHA-256, and cut to a
  safetensors file of those entries (`torch/private/safetensors.rkt`)
- image reading (`torch/vision/image.rkt`, #199): `decode-image` (bytes)
  and `read-image` (a path) to a uint8 `[C H W]` tensor, `#:mode
  'unchanged 'gray 'gray-alpha 'rgb 'rgba`, `#:device`; JPEG and PNG
  through `stb_image` behind `tr_image_decode`, the header from nixpkgs'
  `stb` found by pkg-config like libsndfile and compiled once in
  `src/torchrkt/detail/stb_image.c`; PNGs match
  torchvision.io exactly, JPEGs within a count or two
  (`torch/tests/image-parity-test.rkt`, against torchvision in both shells)
- generative examples on MNIST (#152):
  `torch/scribblings/examples/10-dcgan.rkt` (a DCGAN shrunk to 28x28,
  `ConvTranspose2d` and `BatchNorm2d` in the generator, `leaky-relu` in the
  discriminator, two `adam`s at 2e-4 with
  `#:beta1 0.5`) and `torch/scribblings/examples/11-vae.rkt` (the linear VAE, the reparameterization
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
  fold left); so do `< <= > >=`, which dispatch to lt/le/gt/ge, mirror a
  real on the left, and take two operands with a tensor; `@` is matmul,
  like Python's `a @ b` (`|@|` under the at-reader: lp2 examples and
  Scribble). Write arithmetic, comparisons and matmul with these, not the
  names: `scripts/resyntax-rules.rkt` rewrites `(mul a b)` to `(* a b)`,
  `(ge a b)` to `(>= a b)` and `(matmul a b)` to `(@ a b)`; `t`/`Σ` are terse
  aliases for transpose/sum; unary `T` reverses all axes like Python's `x.T`
  (use `transpose` on the last two axes for batched attention);
  `~> ~>> lambda~> lambda~>>` are re-provided
  from the `threading` library (dep `threading-lib`, prefetched offline by
  the `racket-deps` fixed-output derivation in flake.nix). Thread a chain
  of three or more calls: the same rules file rewrites
  `(clip (+ h (relu (dil h))) l t)` to `(~> (dil h) relu (+ h) (clip l t))`,
  following first arguments and either operand of a two-operand `+` or `*`
  while the other arguments are names or literals (a call that joins two
  computations starts the chain). An expansion analyzer keeps it off a call
  that is already a `~>` or `~>>` step, which receives an argument it does
  not show
- getters: write `device` and `dtype`, not `tensor-device` or
  `tensor-dtype`; `scripts/resyntax-rules.rkt` rewrites the long names
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

**Name shadowing convention:** the 22 ops named like a `racket` export
(`+ - * / < <= > >= abs argmax cos exp flatten length log max min sin sort
sqrt take tanh`) are generic — tensors hit libtorch, anything else defers to
racket's binding — so `(require torch)` never breaks numeric code. New ops
that collide must follow the same dispatch pattern (check all of `racket` —
`tanh`, from racket/math, was missed initially and broke numeric callers),
and docs require `torch/scribblings/labels` for their `for-label` bindings of
`racket`, `torch` and `torch/nn`. `torch/safe` is `torch` minus every name
`racket` exports and `torch/operators` is the rest, both computed with
`subtract-in` (laws in `torch/tests/import-tiers-test.rkt`): a new colliding
op joins `torch/operators` by itself and needs an L6 row. Every other public
module stays disjoint from `racket` (L4), except `torch/generated`, whose
`take` L4 pins.
Dispatching named ops carry dependent (`->i`) contracts so the wrong shape
gets contract blame, not a runtime error; the `+ - * / < <= > >= @` operators are
provided as plain renames (no contract overhead on the numeric fast path),
per `torch/foreign/operators.rkt`.

From `torch/nn`: `define-layer procedure->Layer gen:layer layer? Parameter Buffer LayerList LayerHash parameters
named-parameters in-named-parameters buffers children Linear Conv2d MaxPool2d Flatten Dropout
Sequential Embedding LayerNorm ConvTranspose2d GroupNorm BatchNorm2d BatchNorm1d
LSTM GRU MultiheadAttention TransformerEncoderLayer TransformerDecoderLayer
TransformerEncoder TransformerDecoder sinusoidal-positions causal-mask
layer-copy layer-rebuild
sgd adam rmsprop step! zero-grads! clip-grad-norm! learning-rate
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
mode) and attention (`scaled-dot-product-attention #:mask #:causal?
#:dropout #:scale`, tranche 8, #210, the fused kernel behind
`F.scaled_dot_product_attention`; `gelu #:approximate 'tanh`, GPT-2's form,
on the hand-written shim) live on `torch` beside the other functional ops;
a causal mask built by hand is `(masked-fill scores (eq (tril (ones T T)) 0)
-inf.0)`, whose mask is `#t` = hidden, while the fused kernel's boolean
`#:mask` is `#t` = attend, as PyTorch's function is (the layers'
`#:causal?` flags build the mask themselves;
`torch/scribblings/examples/06-gpt.rkt` is one standard `TransformerEncoder`
call, pre-norm with `#:norm #t`, applied with `#:causal? #t`, #210 L3).
`define-layer` is the Python-style `nn.Module` analog: `#:init` is the
constructor body and assigns declared
fields with `set!`, a field's value classifies it at construction
(`Parameter?`, `Buffer?`, `layer?`, `#f` for absent, anything else plain),
`(with-mode body)` binds `mode` (`'train` or `'eval`, predicates `training?`/`evaluating?`) in `#:forward` to the instance's own mode (`train!`/`eval!` set it and recurse; every layer starts in `'train`),
models are plain struct trees owned by the GC (no global parameter store),
and a layer is called by applying it, `(net x)`, the analog of `__call__`:
`gen:layer` derives `prop:procedure`, so every layer, hand-written ones
included, is applicable, and there is no separate `forward` or callable
`layer-forward` (#235). `LSTM` and `GRU` (`torch/nn/recurrent.rkt`, #153) are
`define-layer` forms whose parameter set depends on `#:num-layers` and
`#:bidirectional?`: `parameters-by-key` registers them under PyTorch's own
names (`weight_ih_l0` .. `bias_hh_l1_reverse`) the way `children-by-key`
registers children, and a rest-argument `#:forward` lets an initial state
follow the input. Applying one answers `(values output h-n [c-n])`. On CUDA
the weights are flattened for cudnn (`cudnn-rnn-flatten-weight`, in place,
the parameters keep their identity) whenever their storage addresses
(`tr_tensor_data_ptr` and `tr_tensor_storage_ptr`, each weight's element and
storage start) differ from those the last flattening left, which catches a
device round trip and a replaced parameter with no move hook, costs `to`'s
identity case nothing, and retries a transient OOM rather than latching it.
Every call checks under a per-layer lock (`torch/private/lock.rkt`: not reentrant,
breakable while waiting, and a waiter takes it over from a killed holder),
never reading the addresses outside it, since a flattening on another thread
rebinds the weights' storage; and a packing, or a refusal, is recorded only
while the weights still lie where it found them and the layer's `#:on-move`
count did not change during it (a round trip can hand back the old
addresses), so a `to` racing the check (which takes no turn) leaves nothing
recorded (#189).
A `#:forward` may declare keyword inputs (`#:kw id` or `#:kw [id
default]`, the default seeing the fields); `gen:layer`'s `prop:procedure`
is a keyword procedure that hands them to a `define-layer` or
`procedure->Layer` forward, while a hand-written `layer-forward`, a generic
method, refuses them. `MultiheadAttention` (`torch/nn/attention.rkt`, #210 L1)
is the first such layer: four `Linear` children `query` `key` `value`
`out` (not PyTorch's fused `in_proj`; `torch/nn/linear.rkt`'s `private` submodule
builds a `Linear` from given tensors so the xavier `[3E, E]` draw splits
without extra RNG use), `#:heads #:dropout #:bias? #:batch-first?
#:key-dim #:value-dim`, applied as `(mha q k v #:key-padding-mask
#:attn-mask #:causal? #:need-weights? #:average-attn-weights?)`; its
boolean masks are `#t` = hidden like `nn.MultiheadAttention`'s (turned
into additive float masks before the fused call), `#:causal?` builds the
causal mask rather than PyTorch's hint, and only `#:need-weights? #t`
leaves the fused kernel, answering `(values output weights)`. Its
application shapes are a per-instance `->i` contract built in `#:init` and
applied with `caller` blame (the shape checks are shared through
`torch/nn/attention.rkt`'s `private` submodule). `TransformerEncoderLayer` and
`TransformerDecoderLayer` (`torch/nn/transformer.rkt`, #210 L2) mirror PyTorch's
children (`self-attn` `multihead-attn` `linear1` `dropout` `linear2`
`norm1..3` `dropout1..3`, `_` spelled `-`) and draw order; `#:heads
#:ffn-width #:dropout #:activation ('relu 'gelu 'gelu-tanh or a procedure)
#:norm-first? #:layer-norm-eps #:batch-first? #:bias?` (`LayerNorm` gained
`#:bias?` for it); applied as `(enc src #:mask #:key-padding-mask
#:causal?)` and `(dec tgt memory #:tgt-mask #:memory-mask
#:tgt-key-padding-mask #:memory-key-padding-mask #:tgt-causal?
#:memory-causal?)`, masks in MHA's sense. `TransformerEncoder` /
`TransformerDecoder` take `#:layers n` and `#:norm` (`#f`, a layer, or
`#t` for a `LayerNorm` of the model width) and three kinds of first
argument: a model width with `#:heads` and the layer keywords (one layer
built, then `n` `layer-copy`s of it: `nn.TransformerEncoder(
nn.TransformerEncoderLayer(...), n)`, seeded parity and stream position
included), a layer (`n` copies of it, the layer itself left out, as
PyTorch's deep copies), or a thunk (called `n` times, each layer drawing
its own values); the stack contract rejects layer keywords or `#:norm #t`
beside a layer or thunk. `layer-copy` (`torch/nn/copy.rkt`) is a deep copy
with no RNG draws: fresh tensors with the same values, device, dtype and
requires-grad (no gradients), children copied recursively, modes kept,
shared layers and tensors kept shared through a memo. It runs on the
`gen:layer` method `layer-rebuild layer child tensor`, which
`define-layer` derives (the generated `rebuild` remakes the struct with
each registered child through `child` and every tensor in its fields,
plain lists, vectors and boxes included, through `tensor`, then
reclassifies; `LSTM`'s plain `entries` list is why plain fields are
walked); a stateless `procedure->Layer` copies, one with state refuses,
and a hand-written `gen:layer` without the method refuses, naming
`layer-copy`. The layer constructor contracts come from one
`transformer/c` macro.
`sinusoidal-positions` (`#:layout 'interleaved | 'halves`, a length or a
position tensor) and `causal-mask` (`#t` above the diagonal, or the float
`-inf` form) live in `torch/nn/positions.rkt`. `clip-grad-norm!` (`torch/nn/clip.rkt`) keeps its scale on the
device. Layer init mirrors
PyTorch RNG consumption (`nn.Linear.reset_parameters`), so a shared
`manual-seed!` yields bit-comparable parameters — the MLP cross-test relies
on this.

**REPL parity.** A tensor prints in the Racket REPL exactly as it does in the
Python REPL — `tensor([[ 1.5410, -0.2934], [-2.1788, 0.5684]])` — via
`prop:custom-write`, reproduced from the data + shape in `torch/foreign/format.rkt`
(PyTorch's `tensor(...)` framing isn't in libtorch's C++ printer). Two
accessors expose the two forms explicitly: `tensor->repr` is the PyTorch repr
(what the REPL shows); `tensor->string` is ATen's C++ `operator<<` text. The
repr reproducer covers the common case (CPU float32, fixed-point, precision 4)
and falls back to the ATen form for scientific-notation values; see the TODO in
`torch/foreign/format.rkt`.

Deferred (see plan): `native_functions.yaml` codegen, `nn.Module` macros, the
broader ATen surface, and the portable raco-catalog candidate story.

## The libtorch source knob

`flake.nix` has `torchSource = "bin" | "python"`:

- **`bin`** (default) — `pkgs.libtorch-bin`, pointed at PyTorch's libtorch
  2.14.0 downloads (nixpkgs still ships 2.9.0); the CUDA build adds cuDNN 9.24,
  which the cu130 download no longer bundles. Small prebuilt download, fast
  cached CI on `aarch64-darwin` + `x86_64-linux`. On `x86_64-linux` the shells'
  Python torch is PyTorch's own wheel of the same release
  (`nix/torch-wheels.nix`): the CPU wheel in `nix develop`, and in `.#cuda` the
  cu130 wheel linked against the CUDA libraries and cuDNN the shim loads. Darwin
  still gets nixpkgs' torch. The cross-test compares to a float tolerance either
  way.
- **`python`** — `pkgs.python314Packages.torch`. Builds against the *same*
  libtorch the parity script imports, so seeded `randn` is **bit-exact** — at
  the cost of a heavy (often uncached on darwin) from-source build. It fails at
  evaluation while that torch is older than 2.14.0, the version the vendored
  schema and the MPS kernels assume.

## Build Commands

[`docs/building.md`](docs/building.md) is the same guide written for people;
a change to a build target or a shell belongs in both. The ordered local loop
for a change, and the gates it has to pass before it is done, are the
`cpp-dev` and `racket-dev` skills under `.claude/skills/`; `.racket-dev.rktd`
declares the Racket gates for the racket-dev plugin (`docs/building.md`,
"Development shells").

```bash
nix build              # builds cpp, installs the pkg, runs raco test + examples
nix build .#cpp        # CMake build + gtest only
nix flake check        # the Nix checks: cpp, format, tidy, line-count, racket,
                       # racket91, racket-floor, matrix-runner-test,
                       # platform-audit-test (Linux), the
                       # racket-deps-*, abi-stamp-* and release-* checks
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

racket scripts/bench/run.rkt micro      # benchmarks, never a gate (#253):
                               # docs/benchmarks/README.md
racket scripts/coverage.rkt --changed   # expression coverage (#173); exits
                               # non-zero below the floor and lists the lines
                               # of the files you touched that no test reaches

scripts/resyntax analyze --local-git-repository . [REF]   # lint gate
                                     # (CI fails on any suggestion; scans
                                     #  changed files, torch/ included; the
                                     #  default rules plus
                                     #  scripts/resyntax-rules.rkt; the
                                     #  literate examples' chunks through
                                     #  scripts/resyntax-lp2.rkt)
scripts/resyntax fix --directory torch
scripts/resyntax-gate [REF]           # CI's lane: red on any suggestion
racket scripts/check-paths.rkt        # named paths are tracked: "Named
                                      # paths" in docs/building.md
raco test --timeout 600 scripts/resyntax-rules-test.rkt \
  scripts/resyntax-lp2.rkt scripts/resyntax-wrapper-test.rkt \
  scripts/check-paths-test.rkt                                 # lint suite tests
raco review torch/**/*.rkt
```

`raco review` does not expand macros, so the pure re-export facades
(`torch/main.rkt`, `torch/foreign.rkt`, `torch/nn.rkt`, `torch/train.rkt`, `torch/safe.rkt`,
`torch/operators.rkt`, `torch/scribblings/labels.rkt`) and `torch/info.rkt` carry a
`#|review: ignore|#` directive.

Resyntax, racket-review and `cover` are pinned, one commit per package, in
`nix/racket-linters.nix`, which also compiles them; shell entry copies them
into `.racket-user` with no network, and the CI lane runs the same trees, so
lint findings change only when the flake does (#63). To bump: change the
package's `rev`, set its `hash` to `""`, `nix build .#racket-linters`, paste
the hash Nix reports, and run the full sweep (`scripts/resyntax analyze
--directory torch`), since CI scans only changed files. Each checkout moves to the new
trees on its next shell entry.

`racket-deps` (flake.nix) holds the catalog deps `torch/info.rkt` declares
beyond the Racket distribution, as source trees only. When those deps change,
or a fresh store reports a hash mismatch (upstream moved), set its
`outputHash` to `""`, `nix build .#racket-deps`, paste the reported hash. The
`racket-deps-current` check says "info.rkt changed; bump racket-deps outputHash",
and also fails on an unmet `#:version` minimum. torch's own platform packages
(`torch-x86_64-linux`, `torch-aarch64-macosx`: `own-platform-packages` in
`nix/racket-deps/racket-deps.rkt`, which must name the same packages as `platform-packages`
in `torch/foreign/raw/native-location.rkt`; `racket-deps-info-test` fails
when they differ) are dropped from the fetch.
Instead, the library-less skeleton under `pkgs/` is installed beside `torch`,
and the staged shim wins over it. `racket-deps.rkt platform-packages
<info-dir> <pkgs-dir> [<platform>]` names the skeletons that apply to the
host (or to `<platform>`), and fails unless each dep's `#:version` and each
skeleton's `version` equal torch's `native-version`; `install-platform <info-dir> <pkgs-dir>
--copy|--link` installs them with `--deps fail`.
Any other `#:platform` dep is refused, whether `torch/info.rkt` or a fetched
package declares it: one hash cannot hold a per-system fetch. So is a
package source (a URL or path): racket-deps fetches catalog packages by
name. `racket-deps-guard-test`
and `racket-deps-info-test` test `nix/racket-deps/racket-deps.rkt`;
`racket-deps-provision-test` tests `nix/racket-deps/provision.sh`, the dev
shells' provisioning. It re-runs when racket-deps, the Racket version or
`torch/info.rkt` changes, runs the same drift check first, removes packages
only the previous racket-deps held, and installs `torch` with `--deps fail`,
never from the live catalog. The flake's install (`nix/matrix/racket-cell.sh`,
which the build, the checks and every matrix cell run) and the shell
provisioning install the host's skeleton first with `install-platform`, so
`--deps fail` finds it.

### The platform package

`pkgs/torch-x86_64-linux/` and `pkgs/torch-aarch64-macosx/` are the
skeletons of the platform packages: each `pkgs/*/info.rkt` (collection
`torch-native`, version equal to torch's `native-version`, which moves only
when the compiled bytes do, apart from torch's `version`), a
`THIRD-PARTY-NOTICES` and the licences only that platform bundles. The
licences both carry (PyTorch's and its third-party code's under
`pytorch-third-party/`, stb's, the codecs') live once, in
`scripts/platform/licenses/`, and `assemble.sh` adds them, and
`torch/LICENSE` as the package's own, to the built package.
`scripts/platform/linux/` builds the Linux libraries outside Nix at the
glibc 2.28 baseline; `scripts/platform/macos/` builds the macOS ones on
GitHub's arm64 runner, since a Linux host cannot build or load them. Both
source `scripts/platform/build-common.sh` (fetching, the codecs,
GoogleTest, the shim's CMake, the gtests) and share
`sources.txt`, `assemble.sh`, `catalog.sh` (whose entries carry each zip's
deps and build-deps), `size-budget.sh` and `smoke-install.sh`, which runs the
installed `torch/pkg-build/smoke.rkt` and checks the modules' package deps;
[`docs/building.md`](docs/building.md#the-platform-package) has the steps.

```bash
scripts/platform/linux/build.sh .scratch/platform     # manylinux_2_28, docker
                                 # --cpus 4: codecs, shim, audit, gtests
scripts/platform/assemble.sh torch-x86_64-linux .scratch/platform/bundle .scratch/pkg
scripts/platform/catalog.sh .scratch/pkg/torch-x86_64-linux.zip .scratch/catalog
scripts/platform/linux/smoke.sh ubuntu:24.04 racket-9.1-x86_64-linux-buster-cs.sh .scratch/catalog
```

`linux/audit-bundle.sh` fails the build on a `/nix/store` reference, a RUNPATH
other than `$ORIGIN`, a missing `$ORIGIN` beside a NEEDED library in the
bundle, a GLIBC, GLIBCXX or CXXABI version above libtorch 2.14's (2.28,
3.4.22, 1.3.11), symbol versions that `objdump -T` and `readelf -V` read
differently, or a NEEDED library outside the bundle and the system set. It
follows NEEDED entries only, so a dlopened library slips past it. The
`platform-audit-test` check is its negative control: the Nix-built shim must
fail it. Every fetched source is pinned by sha256 in `sources.txt`, and the
build's caches are keyed on those sums. The scripts never commit a library.

The macOS package is built only by `.github/workflows/platform-macos-build.yml`,
which `platform-macos.yml` calls, one `macos/build.sh <work> <step>` per step, for the deployment target in
`macos/deployment-target`; [`docs/building.md`](docs/building.md#arm64-macos)
has the triggers, what `macos/audit-bundle.sh` fails on and its negative
control.

`(system-library-subpath #f)` is `aarch64-macosx` on racket-lang.org's
Racket for macOS but `aarch64-darwin` on Nix's. `torch/info.rkt` declares
`torch-aarch64-macosx` once for each string and native-location's
`platform-packages` maps both to it, so the macOS Nix cell installs that
skeleton as the Linux cell installs its own. Likewise a natipkg Racket on
x86_64 Linux, which is what pkg-build.racket-lang.org runs, reports
`x86_64-linux-natipkg`, so `torch-x86_64-linux` is declared for that string
too; Nix never reports it, and no cell provisions for it. Platform deps stay
one string each, never a regexp. Never assume the host's string in a test.

### The build matrix

`nix run .#matrix` builds and tests every cell for the host's system and ends
with a pass/fail/skip table; `nix run .#matrix -- '9.1 zip'` runs only the
cells whose "racket libtorch" contains the filter, and exits 2 when none does.
Its knobs are `MATRIX_CELL_TIMEOUT` (6h a cell), `MATRIX_MIN_FREE_GB` (30),
`MATRIX_LOG_DIR`, `MATRIX_CORES` (4) and `MATRIX_MAX_JOBS` (1). The floor is
Racket 9.1, `("base" #:version "9.1")` in `torch/info.rkt`: 9.0's parallel
threads can lose a `semaphore-wait` result, fixed in 9.1. The `racket-floor`
check holds that floor, the oldest of `racketVersions` and the `racket91`
check's Racket to one release.

### PyTorch parity

The default `nix develop` shell ships a Python with `torch`, so you can explore
PyTorch behaviour beside the Racket bindings and run the real cross-test:

```bash
nix develop --command python3 -c 'import torch; print(torch.__version__)'
nix develop --command raco test torch/tests/python-cross-test.rkt
nix develop --command raco test torch/tests/generated-parity-test.rkt
```

Where python3 can't `import torch` (the sandboxed `nix build`, the lean `.#ci`
shell, the CUDA matrix cell), the torch-oracle tests self-skip. Those three do
carry a bare `python3Minimal`, for `dataset-fixtures-parity-test`: it
regenerates the MNIST and CIFAR-10 fixtures with
`scripts/gen-dataset-fixtures.py` and compares them byte for byte. It skips
only where the generator is absent (an installed package) and fails where the
generator is present but python3 is not.

### Accelerators

`accelerator-if-available` picks CUDA, then MPS, then CPU — what the examples'
`pick-device` returns, and the analogue of PyTorch's
`torch.accelerator.current_accelerator()`, except that it picks MPS only once
one element has been allocated there (#353): GitHub's macos-14 runners
advertise MPS and cannot allocate on it. `mps-available?` keeps PyTorch's
meaning, so the device suite guards its CUDA and MPS bodies separately on the
`-available?` predicates, and on such a host its MPS cases fail; the example
runners train on whichever accelerator the helper returns. Both verify real
hardware where it exists and self-skip where it doesn't — never assume a case
ran because it was green.

**Darwin needs no special shell.** The `aarch64-darwin` `libtorch-bin` ships the
Metal backend, so plain `nix develop` is the MPS verification environment —
there is no `.#mps` counterpart to `.#cuda` (which exists only because CUDA
needs a differently-linked libtorch plus the host driver). Confirm with
`nix develop --command racket -e '(require torch)(mps-available?)'`.

**No MPS kernel gaps.** Since libtorch 2.14 every op the examples use has an
MPS kernel, `aten::_ctc_loss` and `aten::native_group_norm_backward` included
(#139), so `pick-device` returns `accelerator-if-available` unmodified. When a
new op lacks one, or its kernel is wrong for some shapes, route that op alone
through the CPU with `to-device`, which is differentiable both ways, rather
than routing darwin to the CPU. `ctc-loss` does this for targets of 512 labels
or more, where 2.14's MPS backward returns wrong or NaN gradients while the
forward stays right; a green loss curve does not catch that. That routing is
a temporary workaround for an upstream PyTorch bug (#258): remove it, and keep
its regression test, once a pinned libtorch fixes the kernel.

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
- `tests/torchrkt/{random,ops,autograd,generated_golden,generated_tranche2..8}_test.cpp`
  — GoogleTest goldens per family (generated families get a C-boundary
  golden: a correctness case + a null/length-guard case).
  `c_api_compile_test.c` proves the headers are valid C (add a
  function-pointer line for at least one representative of each new op
  family, plus any function whose signature shape is new).
- **Existing C signatures are append-only.** A change to an existing
  `extern "C"` signature, or to a struct or enum that crosses the boundary,
  gets a new symbol instead (`tr_full_on` beside `tr_full_i64_on`), so existing
  callers and the CPU/CUDA stubs in `device.cpp` stay consistent, and a
  compatible-range stamp check stays possible later.
- **The ABI stamp.** `tr_abi_stamp()` is a SHA-256 over every header under
  `include/torchrkt/c_api/` (generated ones included), each named by its
  relative path, in sorted order; `cmake/abi_stamp.cmake` is the one
  computation, run at configure time and by `cmake -P`. The bindings carry a
  copy in `torch/foreign/raw/abi-stamp.rktd`, and the loader refuses a shim
  whose stamp differs. Any edit to a `c_api` header, a comment included, means
  `nix run .#abi-stamp` (`nix run .#codegen` also rewrites it) and committing
  the copy; the `abi-stamp-current` flake check fails until then.

### Racket (`torch/`)

Thin re-export facades over small modules (target ≤ 500 lines/file).

**Import convention:** library modules use `(require (only-in ...))` with
explicit, alphabetized name lists — never whole-module requires — so each
file documents exactly what it pulls in. Exemptions, each marked with a
comment at the require site: pure re-export facades (`torch/main.rkt`,
`torch/foreign.rkt`, `torch/nn.rkt`, `torch/train.rkt`, `torch/safe.rkt`,
`torch/operators.rkt`, `torch/scribblings/labels.rkt`), and macro-heavy modules
whose expansions need the module's full export set (`syntax/parse/pre`).
Runtime paths take `(only-in racket/runtime-path #%datum define-runtime-path)`.

- `torch/info.rkt` — package metadata + native-library pre-install hook.
- `torch/main.rkt` — high-level facade (re-exports `torch/foreign.rkt`).
- `torch/safe.rkt` — `torch` minus `racket`'s names; `torch/operators.rkt` — the rest
  (the import tiers, `scribblings/imports.scrbl`).
- `torch/private/exports.rkt` — `export-names`, a module's sorted export names,
  shared by the tiers test and the imports chapter.
- `torch/foreign.rkt` — the contracted layer + the `unsafe` submodule.
- `torch/data/dataset.rkt` — `define-dataset` and `gen:dataset`;
  `torch/private/definer.rkt` — the clause grammar and field checks it,
  `define-layer` and the trainer definers share.
- `torch/vision/cifar10.rkt` — CIFAR-10 loader and dataset, `vision/fixtures/`
  its 256-record synthetic fixture and MNIST's
  (`scripts/gen-dataset-fixtures.py`), and `vision/fixtures/images/` the
  reader's synthetic JPEGs and PNGs (`scripts/gen-image-fixtures.py`);
  `torch/vision/image.rkt` — the image reader; `torch/vision/diffusion.rkt` — DDPM schedules, `q-sample`
  and the UNet layers; `torch/vision/mnist.rkt` — MNIST loader.
- `torch/text/data.rkt` — character corpora, vocabularies and token blocks;
  `torch/text/translation.rkt` — the seq2seq tutorial's eng-fra pairs;
  `text/fixtures/` their committed excerpts.
- `torch/data/loader.rkt` — `tensor-dataset`, `dataloader`, `in-dataloader`,
  `in-epochs`, re-exporting `torch/data/dataset.rkt`.
- `torch/foreign/ops.rkt` — version/seed + marshalling (`item`, `to-dtype`,
  `uniform!`, `to`); `torch/foreign/creation-ops.rkt` — the constructors
  (`zeros` .. `rand`, `tensor`, `arange`, `eye`, the `*-like` family, with
  placement and `#:requires-grad?` handled once); `torch/foreign/tensor-ops.rkt`
  — the op tranche (and the
  shadow-dispatch convention); `torch/foreign/order-ops.rkt` — `topk` `sort`
  `argsort` `multinomial`; `torch/foreign/autograd-ops.rkt` — autograd +
  `with-no-grad` + in-place ops; `torch/foreign/structs.rkt` — the `tensor`
  wrapper (`prop:cpointer`, shape cached at wrap time, allocator/deallocator
  finalizer); `torch/foreign/error.rkt` — `check-ok` / `check-handle`;
  `torch/foreign/format.rkt` — the PyTorch-repr reproducer.
- `foreign/raw/*.rkt` — direct FFI, one module per C translation unit:
  `syntax` (the FFI definer + `_Tensor` cpointer), `native-location` (the
  one resolver for the shim: the staged `torch/native-libs/` first, then
  collection `torch-native` from the platform package; scripts use its
  `native-library-path`, never a hard-coded `native-libs`),
  `native-library` (opens it and checks the ABI stamp before any binding
  is defined), `fault` (the
  native-fault latch `define-torch` puts under every binding, #76),
  `pressure` (no
  FFI of its own: the collection policy under the ledger, the two troughs
  and the capacity backstop, #145), `pressure-settings` (its parameters;
  `RKTORCH_MEMORY_FRACTION` and `RKTORCH_MEMORY_LIMIT` in MiB set the
  mark's initial values, #236), `device-queries` (the capacity, allocator
  and release queries `device` hands down to it), `race-points` (no FFI:
  the race points between the steps of each critical section, one `unbox`
  when no test hook is installed; a new step in a critical section gets a
  point and a case in `torch/tests/race-*.rkt`, #40), `memory` (the lifetime substrate:
  frees, pressure ledger, `tensor-allocator`, op-definer macros),
  `global`, `tensor`, `random`, `creation`,
  `shape-ops`, `elementwise`, `reduce`, `linalg`, `autograd`.
  **`docs/internals.md` is the canonical memory-management narrative**
  (lifetime chain, phantom-bytes pressure, typed OOM + retry). The
  rules agents must not break: every tensor-returning binding carries
  `tensor-allocator` — or `tensor-allocator/no-retry` for bindings a
  second call would not repeat harmlessly, a draw from the global RNG
  stream (randn/rand) or a tensor-returning op that also updates an
  argument (batch_norm's running statistics; such generated ops are
  flagged `no-retry` in the codegen allowlist, while status-returning
  in-place ops like `add_` take no allocator wrap and no flag), so a
  retry can never double-draw and break seeded parity or apply an
  update twice; `tr_tensor_grad`
  carries `tensor-allocator/gradient`; a binding with several tensor
  outputs carries `tensor-allocator/outputs` (or `/outputs/no-retry`),
  which registers and
  accounts every handle inside one atomic section; never a bare
  `(allocator ...)` wrap (skips the ledger).
  Explicit synchronous release goes through the raising,
  finalizer-cancelling `tr-tensor-free/checked`; OOM reaches users as
  `exn:fail:rktorch:oom` (catch by type, not message). A handler broad
  enough to catch an `invalid memory reference` either lets it through to
  the caller, which latches nothing, or, where no caller is waiting (a
  finalizer guard, the printer, `account!`), calls `note-native-fault!`
  when `native-fault?` holds, so that fault latches the library; never bind
  native code with `define-ffi-definer` or `get-ffi-obj` directly, which
  bypasses the latch.
- `torch/nn.rkt` — pure re-export facade over `nn/` (`torch/nn/layer.rkt` = `gen:layer`, `LayerList` +
  the `define-layer` macro, whose `#:forward` takes a rest argument and
  keyword inputs, whose
  fields admit `parameters-by-key` beside `children-by-key`, and whose
  `#:on-move` body runs after a `to` that rebound any tensor in its subtree,
  judged against placements read before anything moved, so a tensor tied
  across siblings counts for both; its `private` submodule's `define-layer*`
  (under `define-trainer`/`define-callback`, #384) has `set!`-able fields; a
  `set!` raises when the field held a layer, `Parameter`, `Buffer`,
  `children-by-key` or `parameters-by-key` at construction, or when the new
  value is one or holds one in its pairs, vectors, boxes or hash keys and
  values (structs and mpairs are not looked inside); `torch/nn/mode.rkt` = `train!`, `eval!`,
  `call-with-mode` and `in-mode`; `torch/nn/parameter.rkt`, `torch/nn/buffer.rkt`, `torch/nn/linear.rkt`,
  `torch/nn/init.rkt`, `torch/nn/optim.rkt`, `torch/nn/ema.rkt`, `torch/nn/loss.rkt`, `torch/nn/recurrent.rkt`,
  `torch/nn/attention.rkt`, `torch/nn/transformer.rkt`, `torch/nn/positions.rkt`, `torch/nn/copy.rkt`, `torch/nn/clip.rkt`).
- `torch/train.rkt` — pure re-export facade over `torch/train/trainer.rkt`
  (`define-trainer`, `define-callback` and their body forms),
  `torch/train/run.rkt`, `torch/train/loop.rkt` (`fit`, `test`, `predict`),
  `torch/train/logging.rkt`, `torch/train/callbacks.rkt` and
  `torch/train/optimization.rkt`; `torch/train/private/hooks.rkt` is the one
  hook table, read at phase 0 and for-syntax. The loop runs in
  `torch/train/private/fit.rkt` and `torch/train/private/evaluate.rkt`;
  `torch/train/private/dispatch.rkt` calls hooks and holds the failure
  protocol, `torch/train/private/book.rkt` the on-device `log!`
  accumulators, one ledger per loop, `torch/train/private/state.rkt` the run
  struct. Precision, clipping, manual optimization, hparams and checkpoints
  raise `exn:fail:unsupported` until their legs (#385, #386) land. The loops walk a loader with
  `with-traversal` from `torch/data/loader.rkt`'s `private` submodule, which
  stops `#:workers` threads on an early exit.
- `torch/private/download.rkt` — the one download path (#195): a temp file in the
  cache's directory, checked, then installed. `call-with-verified-download`
  checks an exact size and SHA-256 (pretrained weights, hymenoptera);
  `download-cached` hands the downloaded path to a `#:valid?` predicate and
  renames it into place (MNIST, text, translation, CIFAR-10, audio). Each
  loader keeps its own cache-directory function and override variable.
- `torch/private/install-torchrkt-native.rkt` — stages `libtorchrkt.*` into
  `native-libs/` from `TORCHRKT_NATIVE_LIB_PATH` (set by the Nix build/shell).
  Without the variable it prints nothing when `native-location` finds the
  shim, staged or in `torch-native`, and otherwise prints its not-found
  message: the package server fails a package on any stderr from `raco setup`.
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
- `scribblings/examples/` — the literate `scribble/lp2` programs woven into
  the guide; exempt from the import convention.
- `torch/pkg-build/smoke.rkt` — the only test the package build server runs (see
  [The package build server](#the-package-build-server)); its `torch/pkg-build/info.rkt`
  gives it 300 s and omits itself.

### Codegen (`codegen/`)

The ATen generator (v2/A, #2): `nix run .#codegen` (equivalently
`nix develop --command python3 -m codegen`, but with a much smaller
closure) reads `codegen/allowlist.txt` against the **vendored** schema in
`codegen/aten/` (pinned to the C++ libtorch 2.14.0 — see the README there;
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
  battery in `torch/tests/generated-parity-test.rkt`; every new allowlist line needs an
  input recipe in that test (`'device-only` for an op with no CPU kernel,
  which then needs a device-guarded test of its own)

Conventions:

- **Extend the allowlist instead of hand-writing** when an op fits the IR
  (Tensor / Scalar→double / int64 / bool / IntArrayRef / TensorList args,
  one or more Tensor returns). Unsupported signatures are skipped with a report —
  widening the IR is a generator change, not a hand-written shim.
- Optional *types* are in the IR: `Tensor?` is a NULL pointer, `int?`,
  `Scalar?` and `float?` carry a presence flag, `int[]?` a length plus flag, and
  `ScalarType?` a -1 sentinel, and `Generator?` a `tr_generator` handle
  (NULL for the global stream; an op that draws still needs the allowlist
  `no-retry` flag); `Tensor?[]` (#365, `index.Tensor`) is the TensorList
  encoding whose NULL entries are `None`. In-place ops (`add_`) emit a mutable receiver
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
examples (`torch/scribblings/examples/*.rkt`, prose-by-design) and the
python parity twins (`examples/python/*.py`, whose docstrings identify the
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
module reaches the definition while `torch/foreign.rkt` requires
`(submod ... checked)` and re-exports it.  Without the split, contracting
`add` would contract `torch/foreign/operators.rkt`'s `t+`, and contracting `tensor?`
would make every contract in `torch/foreign/contracts.rkt` cross a second boundary.

Which form to use is decidable, not a judgement call: **does a module under
`torch/foreign/` import this name?**  If yes `define/checked-out`, if no
`define/contract-out`.  A test in `torch/tests/contract-out-test.rkt`
asserts no module under `torch/foreign/` requires a `checked` submodule, so
the fast path cannot be recontracted by a later edit.  The same rule and
the same test cover `torch/nn/`, with `torch/nn.rkt` as the facade.

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
the `struct`.  `torch/foreign.rkt` is a pure re-export module: it requires each
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

`racket scripts/bench/run.rkt contracts`, lab host, rounded.  The tensor
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
on the hot tensor surface are the expensive part of `torch/foreign.rkt`.

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
(regenerate + fail on porcelain).
`platform-linux-build.yml` proves the Linux zip builds, audits, passes its
gtests and does not exceed its size budget. `platform-linux.yml` proves it
installs and passes the smoke on ubuntu-24.04 and `rockylinux:8`, and that the
manual renders with all its deps declared. `platform-macos-build.yml` proves
`torch-aarch64-macosx` builds, audits, installs and passes the smoke on
`macos-latest`; `platform-macos.yml` calls it and proves the zip passes the
CPU suites on `macos-14` (whose runners advertise MPS but cannot allocate on
it), and, weekly or when asked, that its MPS tests pass. `pkg-build.yml` proves the package build server's own tool accepts the
Linux zip (see [The package build server](#the-package-build-server)).
`release.yml`, on a `v*` tag, checks the versions and the ABI stamp
(`scripts/release/`, tested by the `release-test` and `release-current`
checks), calls both platform lanes when the natives changed, and drafts the
GitHub release; `release-canary.yml` installs `torch` from the catalog
weekly. [`docs/releasing.md`](docs/releasing.md) is the runbook. The
job-by-job detail is in
[`docs/building.md`](docs/building.md#continuous-integration-and-the-binary-cache).

### The package build server

pkg-build.racket-lang.org is a natipkg Racket (`x86_64-linux-natipkg`, so
`torch/info.rkt` declares `torch-x86_64-linux` for that string too) in a
1 GB VM. Its test column is `raco test --drdr --package torch`: one process
per file, parallel to the processor count, any stderr a failure, 90 s a file
and **ten minutes for the whole run, a hard limit**. It sets
`PLT_PKG_BUILD_SERVICE` (and lists it in `PLT_INFO_ALLOW_VARS`, without which
`#lang info`'s `getenv` sees nothing); key on that one, never `CI`, which
GitHub sets too. With it set, `torch/info.rkt`'s `test-omit-paths` omits
every top-level entry but `pkg-build/`, so the server runs only
`torch/pkg-build/smoke.rkt`: the `test` submodules of a few fast suites,
04-mlp and a FLAC round trip, loading libtorch once. Keep it cheap and
silent on stderr; it runs in the normal suite as well. It requires torch by
collection, never by relative path: `nix flake check` tests the source tree
beside a `--copy` install, where `"../main.rkt"` is a second instance whose
devices 04-mlp's contracts reject.
`torch/tests/server-test-set-test.rkt` asks `raco test` itself which files the
server would run and fails on anything but the smoke, so a new top-level
entry in `torch/` must join the omit list. The docs column renders the manual
with libtorch loaded, also within the 1 GB.
[`docs/building.md`](docs/building.md#reproducing-the-package-build-server)
has the harness's knobs.

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

## Before a stack is ready for review

CI never trains past a few fixture steps on a CPU, so it cannot see a change
that keeps an example running but stops it learning. Before a stack is marked
ready for review, and before a release, run the epoch check at its top commit
on the lab host:

```bash
nix develop .#cuda -c scripts/epoch-check
```

It trains every example for a few epochs and compares each headline with the
record in `docs/epoch-check.md`; its table goes in the top pull request's
review packet. A `REGRESSION` or `FAILED` row blocks the stack until it is
explained, and a change that moves a number on purpose updates the record in
the same pull request.

## Code Review Rules

Codex reads this section; the Claude review workflow gives its bot
master's copy.

### Stacked pull requests

- A pull request whose base is another pull request's branch is one layer
  of a stack, and what it changes is its diff against that base. The
  lower layers' code is already on the base branch and is reviewed on
  their own pull requests: do not flag what that code already did, and do
  not ask for anything a lower layer provides, but do flag what this
  layer's diff breaks in another layer's code.

### Generated code

- The generator's output is the files it emits, each with an
  `@generated by codegen ... DO NOT EDIT` header:
  `cpp/include/torchrkt/c_api/generated.h`, the `.h` files in
  `cpp/include/torchrkt/c_api/generated/`, the `.cpp` and `.cmake` files
  in `cpp/src/torchrkt/generated/`, `torch/generated.rkt` and
  `torch/tests/generated-parity.rktd`. Review the generator
  (`codegen/*.py`) and `codegen/allowlist.txt` instead of their bodies;
  when a template changes, check one emitted body per changed template
  path against it. When the allowlist gains an op, check its emitted C
  signature, Racket binding and allocator wrap: a new schema shape can
  reach a classifier path no earlier op exercised, and codegen drift only
  proves the output is reproducible. Any other file is reviewed like any
  other, header or not.
- The vendored schema under `codegen/aten/` is upstream data: when it
  changes, check its sha256 against the pin in `codegen/aten/README.md`
  and review the emitted bodies of allowlisted ops whose schema changed,
  not the YAML itself.

### What matters most

- Native memory across the FFI boundary: the allocator wrap on every
  tensor-returning binding, with its no-retry and gradient variants
  (Architecture > Racket, `foreign/raw/*.rkt`); the `detail/op_call.hpp`
  helpers every C++ op body reduces to (Architecture > C++).
- PyTorch parity: an allowlist op needs a recipe in
  `torch/tests/generated-parity-test.rkt`, and a `'device-only` one its own
  device-guarded test; a hand-written op needs a python cross-test check
  beside its gtest golden; a new example comes with its runner
  (`examples/test/`), a row in `docs/epoch-check.md`, and a Python twin,
  either in `examples/python/` or as a parity suite under
  `torch/tests/python/` (as 16-style-transfer's is).
- Conventions: contracts at the definition site ("Validating arguments");
  an op named like a racket/base, racket/list or racket/math function
  defers to it for anything but a tensor; Racket modules within the
  500-line target.

Formatting, clang-tidy, Resyntax, the C++ 500-line gate and codegen drift
are CI's checks; leave them to CI.
