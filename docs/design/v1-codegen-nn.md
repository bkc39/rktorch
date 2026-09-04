# rktorch v1 — ATen binding strategy & nn.Module architecture

Status: **decided / implemented**. The nn.Module gate at the end is closed:
**Architecture 1** (Python-style `define-module` macro over a `gen:module`
generic interface, module-tree parameter registry, no VarStore). See *Open
decision gate* at the end for the recorded rationale.

Companion: `plans/v0-scaffold.md` (what shipped).

---

## 1. Context — where v0 left us

v0 ("rktorch v0 scaffold") shipped the **spinal cord, not the API**: the full
dev-ops suite ported from the sibling FFI repos (`xgboost-rkt`, `scs`, `glmnet`)
— Nix flake with a `torchSource = bin | python` knob, CI matrix on linux +
darwin, resyntax + clang-tidy + line gates, and a `python-cross-test.rkt` parity
harness that **SKIPs cleanly when `import torch` fails** — proven by a single
vertical slice: `manual-seed!` + `randn` (fixed 2×2) + tensor print.

The substrate that matters is already in place:

- opaque **`Tensor*` handle** with a **GC finalizer** (`at_free`-style),
- the **`tr_last_error`** exception-to-status contract,
- a CMake-built `libtorchrkt.{dylib,so}` **staged into the Racket runtime** by
  `torchrkt/private/install-torchrkt-native.rkt`.

v0 **explicitly deferred** to v1: a usable ATen surface, **autograd**, the
`nn.Module` system, and any binding **codegen**. This document settles two of the
forks the original brain-dump (`torch-plan.md`) raised — how ATen ops get bound,
and how the module system is shaped — grounded in how every other serious
libtorch binding does it. v1's concrete goal: **tensor tranche + autograd + one
`Linear`/`MLP`, validated against PyTorch.**

---

## 2. ATen op bindings — decision: hand-write the shim for v1

Two questions were conflated ("does the C++ really need to be *emitted*? can't I
just write a C++ lib built with CMake and stage it like the other shims?").
Untangled:

### 2.1 Do we need a C++ shim at all? Yes — unlike the other repos.

xgboost exposes a C API (`XGBoosterPredict`), SCS exposes a C API, glmnet is
C-callable Fortran — so their Racket FFI binds them **almost directly**.
**libtorch has no C ABI.** ATen is C++: `at::Tensor torch::randn(...)`, heavily
overloaded, templated, throwing C++ exceptions, taking C++ types (`at::Tensor`,
`c10::optional<T>`, `at::IntArrayRef`, `at::Scalar`, `c10::Device`,
`at::ScalarType`). Racket's FFI cannot call that — no stable C++ ABI, names are
mangled, overloads don't exist at the ABI level. A thin **`extern "C"`
translation layer is mandatory**: it turns each C++ op into a flat C function
over opaque `Tensor*` handles, catches C++ exceptions, and reports them via
`tr_last_error`. That is "the C++ side," and v0 already has its skeleton.

### 2.2 Generate that shim, or hand-write it? Hand-write it for v1.

Generation ("emit") only earns its keep at the **full ATen surface** (~2,000 ops
× hundreds of overloads), where the per-op marshalling boilerplate is the entire
cost — which is exactly why tch-rs / ocaml-torch / hasktorch / R-`lantern` run a
generator (`gen.ml`) over PyTorch's `Declarations.yaml`. For a **curated v1
tranche of a few dozen ops**, hand-writing both sides is simpler, more
debuggable, and matches the exact pattern already used in xgboost/scs/glmnet. The
generator is a **v2 lever** — documented and deferred — to pull when hand-writing
becomes the bottleneck.

### 2.3 How every serious libtorch binding does ATen ops (reference)

| Lang | Project | ATen op codegen | nn strategy |
|---|---|---|---|
| Python | torch (reference) | `torchgen` over `native_functions.yaml` | pure-Python `nn.Module` |
| C++ | libtorch frontend | n/a (native) | `torch::nn` directly (the only one) |
| Rust | **tch-rs** | `gen.ml` over `Declarations.yaml` → `torch_api_generated.{cpp,h}` + `torch-sys` | VarStore + `Module` trait (host) |
| OCaml | **ocaml-torch** | same `gen.ml` (tch-rs's "comes from ocaml-torch") | `Var_store` + `Layer` (host) |
| Haskell | **hasktorch** | codegen over Declarations.yaml; raw `libtorch-ffi` + typed layer (type-level shapes) | typeclass over a record (host) |
| .NET | **TorchSharp** | generated + hand C++ shim | mirrors `torch.nn` in C# via reflection |
| Go | **gotch** | tch-rs-style Declarations.yaml C-API | host structs |
| Ruby | **torch.rb** | generated from `native_functions.yaml` via Rice | mirrors Python in Ruby |
| R | **torch (mlverse)** | C-ABI wrapper **`lantern`** auto-generated | nn_module reimplemented in R (R6) |

**The universal substrate — v0 already conforms:**

1. Codegen (when present) is **ahead-of-time**, emitting *committed* source —
   never a runtime YAML read.
2. A flat **`extern "C"` wrapper** over C++ ATen; ops take/return opaque `tensor`
   = heap `torch::Tensor*`, results via out-pointers.
3. Tensors freed by a **host-language GC finalizer** calling `at_free`.
4. C++ **exceptions caught at the boundary** and stashed (`PROTECT` →
   `get_and_reset_last_err`); v0's `tr_last_error` is exactly this.

rktorch is already on the canonical path. v1 just (a) widens the hand-written
tranche, (b) adds autograd, (c) adds a host-side module system.

---

## 3. nn.Module — strategy deep-dive

**The framing fact:** a `Module` is *just a named container of parameters +
buffers + submodules, plus a `forward` function.* All the real work is
**autograd** (`requires_grad`, `backward()`, `.grad`, optimizer `step`); the
module layer is bookkeeping. So **v1 exposes autograd first**, and every option
below is a different way to organize that bookkeeping.

> Crucially — **even the C++ `GenericModule` option still computes `forward` in
> Racket**, because `forward` *is* your model code; C++ can only hold the
> parameter registry. This is the single most important thing to internalize
> before choosing.

The field splits into four architectures. For each: what it is, who uses it, and
what it looks like **in Racket**.

### Architecture 1 — Class + auto-registration (Python, TorchSharp, R, Ruby)

**Python (the reference)** overrides `__setattr__`; assigning a
`Parameter`/`Module`/buffer as a field auto-registers it into ordered dicts
`_parameters` / `_modules` / `_buffers`. `forward()` is a normal method;
`.parameters()` recurses through `_modules`; `state_dict` walks the same dicts.

```python
class MLP(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.fc1 = nn.Linear(i, h)   # __setattr__ registers a submodule
        self.fc2 = nn.Linear(h, o)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))
```

- **TorchSharp** reproduces this in C# via `RegisterComponents()` + **reflection**.
- **R (mlverse)** via **R6** classes + an active-binding `set` hook
  (`self$fc1 <- nn_linear(...)`).
- **Ruby (torch.rb)** ~1:1 with an overridden setter.

**Racket version (recommended).** A `define-module` macro that **knows the field
list at expansion time** — so registration needs *zero runtime reflection*
(cleaner than Python's `__setattr__` hack). Params/buffers/submodules live in the
module struct (Python-style "module tree"); `(parameters m)` recurses to feed the
optimizer.

```racket
(define-module linear (in out)
  #:params    ([w (kaiming-uniform (list out in))]
               [b (zeros (list out))])
  #:forward   (x)
  (+ (matmul x (transpose w 0 1)) b))

(define-module mlp (in hidden out)
  #:submodules ([fc1 (linear in hidden)]
                [fc2 (linear hidden out)])
  #:forward    (x)
  (fc2 (relu (fc1 x))))

;; usage
(define net (mlp 784 128 10))
(define opt (sgd (parameters net) #:lr 0.1))   ; parameters recurses the tree
(define yhat (forward net x))                  ; or (net x) via prop:procedure
```

*Under the hood,* `define-module` expands to: a struct with one field per
param/buffer/submodule; a constructor that allocates param tensors with
`requires_grad #t` and builds submodules; a `forward` method; and a `parameters`
method that recursively flattens. Optionally `prop:procedure` so `(net x)` works
like Python's `__call__`.

- **Pros:** matches the `class MyModule(nn.Module)` mental model exactly; no
  global mutable store; submodule nesting + `(parameters net)` "just work";
  statically-checked fields; C++ stays op-only.
- **Cons:** a non-trivial `syntax-parse` macro to write/maintain; param init,
  `state_dict`, `to(device)` are our code, not borrowed from C++.

### Architecture 2 — Explicit VarStore (Rust tch-rs, OCaml ocaml-torch)

No class magic. A `VarStore` owns *all* tensors; a `Path` is a scoped name
prefix; a layer constructor creates parameters **through the path** (inserting
them into the store) and returns a plain struct/closure with a `forward`. The
optimizer takes the whole store.

```rust
let vs = nn::VarStore::new(Device::Cpu);
let net = nn::seq()
    .add(nn::linear(&vs.root() / "fc1", 784, 128, Default::default()))
    .add_fn(|x| x.relu())
    .add(nn::linear(&vs.root() / "fc2", 128, 10, Default::default()));
let mut opt = nn::Sgd::default().build(&vs, 0.1)?;   // optimizer reads the store
```

**Racket version.**

```racket
(define vs (make-var-store #:device 'cpu))
(define (linear path in out)
  (define w (vs-param vs (path/ path "w") (kaiming-uniform (list out in))))
  (define b (vs-param vs (path/ path "b") (zeros (list out))))
  (lambda (x) (+ (matmul x (transpose w 0 1)) b)))
(define fc1 (linear (vs/ vs "fc1") 784 128))
(define fc2 (linear (vs/ vs "fc2") 128 10))
(define (net x) (fc2 (relu (fc1 x))))
(define opt (sgd (var-store-trainables vs) #:lr 0.1))
```

- **Pros:** simplest *runtime* model (a name→tensor dict); device-move,
  `save`/`load`, and "give the optimizer everything trainable" are trivial and
  centralized; no macro needed; battle-tested by the two closest analog
  languages.
- **Cons:** **string path-naming friction** (you manage `"fc1"/"w"` keys); the
  store is ambient mutable state threaded everywhere; further from the
  `class MyModule` mental model.

### Architecture 3 — Typeclass/interface flattening a record (Haskell hasktorch)

The **record is the module**; a typeclass (`Parameterized`, often via GHC
Generics) flattens it to `[Parameter]`; a `Randomizable` spec builds initialized
params. Typed hasktorch additionally tracks shapes at the type level.

**Racket version.** Define a `gen:module` generic interface (`module-forward`,
`module-parameters`); a layer is a plain struct that implements it;
`module-parameters` recurses structurally. This is Architecture 1 *without the
macro* — you write the struct + interface methods by hand.

```racket
(struct linear (w b)
  #:methods gen:module
  [(define (module-forward self x)
     (+ (matmul x (transpose (linear-w self) 0 1)) (linear-b self)))
   (define (module-parameters self) (list (linear-w self) (linear-b self)))])
```

- **Pros:** maximally explicit and macro-free; very Racket-idiomatic (generic
  interfaces); easy to reason about.
- **Cons:** boilerplate per layer (the Arch-1 macro exists precisely to remove
  this); no auto field discovery — you list params/submodules by hand.

### Architecture 4 — C++ GenericModule / TORCH_MODULE (only the C++ frontend does it)

A generic C++ `torch::nn::Module` holds param/buffer/submodule lists; Racket
drives it over FFI (`tr_module_new`, `tr_module_register_parameter`, …);
`TORCH_MODULE` wraps it in a `shared_ptr` holder.

**The catch that sinks it for us:** `forward` *is your model code* — it can't be
generic C++. So a `GenericModule` can only own the **parameter registry**; every
`forward` still runs in Racket over tensor ops. You'd pay FFI chatter on every
registration and serialization round-trip to gain… a parameter dict you already
get for free in Racket. **This is why no scripting-language binding (Python, R,
Ruby, Rust, OCaml, Haskell) takes this route** — only the native C++ frontend,
where `forward` genuinely *is* C++.

- **Only real upside:** direct reuse of C++ **optimizers** and TorchScript
  **save/load** that expect a real `torch::nn::Module`.
- **Verdict:** not recommended as the primary design. If C++-optimizer or
  TorchScript-serialization compatibility is ever needed, add a *thin* C++ module
  object **behind** Architecture 1/2 (a "hybrid"), not as the front door.

### Recommendation

**Architecture 1 (Python-style macro + module-tree params)** — closest to the
`class MyModule(nn.Module)` mental model, keeps C++ op-only, and makes submodule
nesting + `(parameters net)` ergonomic. **Architecture 2 (VarStore)** is the
strong runner-up and is *less work to ship first* (no macro). They are **not
mutually exclusive**: ship the VarStore runtime first, then layer the
`define-module` macro on top of it later, getting both.

---

## 4. What v1 builds (phased)

Each phase is independently green in CI. Example-driven per `torch-plan.md`:
every op/feature lands with an `examples/python/NN_*.py` +
`examples/racket/NN-*.rkt` pair and a tolerant parity test.

1. **Tensor tranche** (hand-written shim widening v0): creation
   (`zeros/ones/full/arange/eye/randn`, `tensor` from a flat vector + shape),
   shape (`reshape/view/transpose/permute/squeeze/unsqueeze/cat/stack`),
   elementwise (`add/sub/mul/div/neg/exp/log/sqrt/pow`, `relu/sigmoid/tanh`),
   reductions (`sum/mean/max/min/argmax/softmax/log_softmax`), linalg
   (`matmul/mm/mv/dot`), and out-marshalling (`tensor->list`/`->f64vector`,
   `item`, `shape`/`numel`, `to`). Keep v0's handle + finalizer + `last_error`
   verbatim; add gtest goldens for representative ops.
2. **Autograd** (prerequisite for nn): `requires_grad_`, `backward`, `grad`,
   `detach`, a `with-no-grad` scope, and in-place `sub_`/`zero_`/`mul_`. SGD can
   be pure Racket (under `with-no-grad`: `p.sub_(lr * p.grad)`; zero grads).
   Parity: `y = x²` → `grad == 2x`.
3. **nn.Module system** (architecture per §3 gate) in `torchrkt/nn/`
   (`module/linear/init/optim.rkt`): recursive `parameters`,
   `forward`/`prop:procedure`, init (`kaiming-uniform`, `zeros`), `sgd`
   (`step`, `zero-grad`); first layer `linear`, first model `mlp`, loss
   `mse-loss` (and/or `cross-entropy`).
4. **End-to-end proof + parity:** matching `NN_mlp.py` / `NN-mlp.rkt` — same
   model/seed, one optimizer step, compare initial + post-step params + loss
   within tolerance via the v0 cross-test harness (SKIP out-of-shell). Stretch:
   a few steps and assert loss decreases.

**v2 lever (deferred):** a generator over `native_functions.yaml` emitting the
broad ATen `extern "C"` wrappers + Racket FFI, to be pulled when the
hand-written surface gets large.

---

## Open decision gate (before Phase 3) — CLOSED

**Decision (2026-06-10): Architecture 1**, expanding into Architecture 3's
`gen:module` interface (so hand-written layers implement the interface
directly, and `define-module` is sugar over it). **The VarStore (Arch 2) is
dropped entirely**, not staged first: the maintainer's constraint that tensor
lifetime be owned by the Racket GC is exactly what a central store defeats —
it roots every parameter, turning lifetime back into explicit store
management. The module tree *is* the store; drop the model and the v0
finalizers reclaim every native handle.

Implementation: `torchrkt/nn/module.rkt` (interface + macro),
`nn/{init,linear,optim,loss}.rkt`, facade `torchrkt/nn.rkt`. Registration is
compile-time (the macro knows the field list), so there is no runtime
reflection at all — strictly less machinery than Python's `__setattr__` hook.

**Revised (2026-09, #97): registration moved to construction time.** The
compile-time field list left nowhere to write ordinary constructor code
(`#:coerce` grew to fill the gap) and could not express `Sequential`'s
indexed children or a `#f` "declared but absent" parameter. `define-layer`
now takes an `#:init` body that assigns declared fields with `set!`, and the
*value* a field holds when the body finishes classifies it: `Parameter?` and
`Buffer?` are tensor subtypes (`torch/nn/parameter.rkt`), `layer?` is a
child, `#f` is absent, anything else is a plain field. `LayerList` holds a
variable number of children under indexed names, with `#:prefix` for a
container that names them without its own field segment. Architecture 1's
other commitments stand: the model is still a GC-owned struct tree, the
`gen:layer` interface is still what hand-written layers implement, and
parameter order (own first, then children, in declaration order) and the
dotted names are unchanged, so checkpoints and the seeded parity tests carry
over. The form was renamed from `define-module` at the same time, since
`module` is a core Racket word (#97).
