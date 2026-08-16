# GPU / native memory management via the Racket GC

Design for #37 (GC pressure accounting) and #38 (OOM error-path
hardening), plus the observability the fixes need. Grew out of the
train-gpt >40-epoch crash investigation (probe data on those issues,
2026-08-02/03).

## The problem, as measured

Racket only sees a tensor as a tiny wrapper struct, so its GC feels no
pressure while native buffers pile up awaiting finalizers:

- Churning 2 GB of immediately-dropped 4 MB tensors leaves **all 2 GB
  in RSS** (264 → 2267 MB) — zero finalizers run during the loop.
- At train-novel scale the same lag parks dead intermediates on the
  card between incidental majors; on a busy GPU (the reported crash ran
  beside ~18.7 GiB of other jobs) one failed allocation follows.
- When that failure lands, `tr_tensor_free`'s bare `delete` can throw
  across the FFI line *inside a GC finalizer* on the errored CUDA
  context → the "invalid memory reference" handler cascade → host OOM
  kill. (Balloon repros produce clean single exns only because exit
  skips pending finalizers.)

## The mechanism, validated

`make-phantom-bytes` is the Racket-CS-supported way to charge native
allocations to the GC:

- `(make-phantom-bytes n)` exists on Racket 9.2; 100 × 10 MB phantoms
  move `current-memory-use` 97 MB → 1.1 GB — the GC counts them.
- A/B on real tensor handles (phantom in a weak hasheq keyed by the
  handle): the same 2 GB churn retains 1.35 GB high-water instead of
  everything — pressure triggers majors mid-loop, majors run
  finalizers, finalizers free native buffers. The residual high-water
  is Racket's normal heap-growth trigger, tunable later if needed.

## Design: three legs, three PRs

### Leg 0 (#38, first, small): noexcept boundary hardening

- `tr_tensor_free`: `try { delete t; } catch (...) {}` — a finalizer
  has nowhere to report; swallowing is correct. This is the one
  boundary function violating the "nothing throws across the FFI line"
  convention (assumed infallible; CUDA free paths can throw after a
  context error).
- Audit the other `void`-returning boundary fns (grad-mode/device
  setters) for the same guarantee.
- gtest: none can exercise a poisoned CUDA context portably; the
  noexcept guarantee is the deliverable. Standard cpp-dev loop.

### Leg 1 (#37, the centerpiece): phantom-bytes accounting

C surface (one new probe, standard three sync points + gtest):

    int tr_tensor_nbytes(const tr_tensor* t, int64_t* out);

`nbytes` (= numel × itemsize of the *view*) rather than storage bytes:
two views over one storage each charge their own extent — mild
over-counting is safe for pressure; a narrow view pinning a large
storage under-charges, which is acceptable approximation (documented).

Racket side — one choke point in `torch/foreign/raw/syntax.rkt`:

    ;; replaces every bare (allocator tr-tensor-free/raw)
    (define tensor-allocator
      (let ([phantoms (make-weak-hasheq)])
        (lambda (raw-fn)
          (define wrapped ((allocator tr-tensor-free/raw) raw-fn))
          (lambda args
            (define t (apply wrapped args))
            (when t (hash-set! phantoms t (make-phantom-bytes (nbytes t))))
            t))))

- Weak keys: when the handle dies, the entry drops, the phantom
  collects, the charged pressure vanishes. No lifetime coupling beyond
  what the allocator wrap already established.
- `nbytes` failures (rc≠0) charge 0 and proceed — pressure is
  best-effort, never a new failure mode on the allocation path.
- Hand-written raw modules switch wraps mechanically; the codegen
  emitter (`codegen/emit_racket.py`) emits the same wrap for generated
  bindings + regenerate (codegen-drift keeps it honest).
- Device policy: charge CUDA bytes 1:1 alongside host bytes initially
  (pressure just needs to scale with total native footprint to make
  finalizers timely); revisit weighting only if measurement demands.

### Leg 2: observability + good-neighbor knobs

- `tr_cuda_memory_allocated` / `tr_cuda_memory_reserved` (caching-
  allocator stats): lets tests assert per-process VRAM without
  nvidia-smi (whose board totals contaminated the first investigation).
- `tr_cuda_empty_cache`: return reserved-but-unused blocks to the
  driver — scripts call it after training so a finished process isn't
  squatting on reservation high-water.

### Explicit non-goals (for now)

- No GC-and-retry on allocation failure inside the shim (re-entrant
  callback into Racket from C — complexity not justified once pressure
  keeps the footprint near the working set).
- No per-storage accounting, no custodian scoping, no allocator
  configuration surface (`PYTORCH_CUDA_ALLOC_CONF` stays an env-var
  note in docs).

## Validation gates (all pre-existing probes)

1. Churn A/B (above): high-water bounded with the wrap in place, no
   change to results.
2. mem-probe at train-novel scale on an idle card: per-process VRAM
   (now via `tr_cuda_memory_allocated`) flat AND ≈ working set.
3. The 22 GiB-balloon training run survives (or fails with one clean
   exn post-#38 — never the cascade).
4. Full parity suites green (accounting draws no RNG and must perturb
   no values): 273941 checks under the cuda shell.
5. Op-throughput microbench (100k small elementwise ops) before/after —
   the weak-hash insert + phantom alloc per tensor must stay in the
   noise.

## Sequencing

| PR | Content | Size |
|----|---------|------|
| A | #38 noexcept free + void-fn audit | small, cpp only |
| B | #37 nbytes probe + tensor-allocator wrap + codegen | medium, cpp + racket + codegen |
| C | cuda memory stats + empty-cache + script/docs wiring | small |

A lands first (kills the crash class); B makes the footprint honest;
C makes both verifiable in-repo.
