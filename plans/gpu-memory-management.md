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

### Leg 0 (#38, first, small): finalizer hardening — at the Racket layer

- Empirical finding (finalizer_death_test.cpp, a throwing from_blob
  deleter freed through the boundary): a throw during storage release
  unwinds through libtorch's own implicitly-noexcept frames
  (TensorBase's noexcept move-assign, ~TensorImpl/~StorageImpl/
  ~DataPtr) and reaches std::terminate before ANY C++ catch at our
  layer — two catch-based revisions of tr_tensor_free both still
  aborted the child. C++-side hardening is impossible here.
- So: `tr_tensor_free` stays a bare `delete` with the honest comment;
  the death test pins the terminate behavior (EXPECT_DEATH) and flips
  loudly if a libtorch upgrade ever makes the release path catchable.
- The live guarantee is Racket-side: raw/syntax.rkt wraps the
  deallocator (`with-handlers exn:fail? → void` around the C call), at
  the single choke point every allocator wrap references. This
  swallows the failure class actually observed in the #38 cascade —
  faults the Racket runtime converts to exceptions — right at the
  finalizer boundary, so error handlers never loop.
- Honest scope: a genuine std::terminate/SIGABRT or unrecoverable
  fault still kills the process — no layer can catch that — but it is
  one clean death, not a cascade, and the caching allocator's normal
  free path (block returned to pool, no CUDA call) makes it rare.
- Audit result: tr_tensor_free is the only void boundary fn.

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

### Leg 1.5: graceful OOM — typed errors + collect-and-retry

Failing *cleanly* (leg 0) is table stakes; failing *gracefully* needs
two more pieces:

**Typed OOM errors.** libtorch throws `c10::OutOfMemoryError` (a
distinct subclass) for allocation exhaustion; the shim's catch-all
currently flattens it into the generic message string. Add a kind
channel beside `tr_last_error`:

    int tr_last_error_kind(void);   /* 0 generic, 1 out-of-memory */

set by every boundary catch (`catch (const c10::OutOfMemoryError&)`
before the general handler). The Racket error path raises a
distinguishable exn (`exn:fail:rktorch:oom`, a struct subtype of
`exn:fail`) so callers catch OOM by type, not by regexing messages.

Backend matrix for the classifier (probed on libtorch 2.9):

- **CUDA**: caching allocator throws `c10::OutOfMemoryError` ("CUDA out
  of memory ...") — the type-catch handles it.
- **CPU**: `alloc_cpu.cpp` fails via the caffe2-style enforce ("[enforce
  fail at alloc_cpu.cpp] ... DefaultCPUAllocator: can't allocate
  memory ... Error code 12") — plain `c10::Error`, NOT
  OutOfMemoryError, so the classifier needs a second, contained match
  for this shape or the CPU case silently degrades to generic-kind and
  skips the retry where it works best. Bonus: CPU OOM is portably
  provokable (one absurd request), so leg 1.5 gets a real gtest
  asserting the classification — the regression guard the CUDA path
  can't have. Caveat: gradual host exhaustion under Linux overcommit
  arrives as the OOM killer (SIGKILL), not an exception; leg 1's
  pressure is the defense there, not this leg.
- **MPS**: not on the device surface today (darwin CI runs CPU). When
  added: MPS OOM has its own c10 error ("MPS backend out of memory" +
  high-watermark numbers) — verify its type then; unified memory means
  phantom-bytes accounting maps 1:1 (GPU bytes ARE host bytes), and
  the finalizer hardening is already backend-agnostic.

**Collect-and-retry at the allocation choke point.** Purely
Racket-side — no re-entrant C→Racket callback needed: when a raw call
returns NULL and the error kind is OOM, the wrapper runs
`(collect-garbage)` (finalizing dead handles returns their blocks to
the caching allocator), optionally `tr_cuda_empty_cache`, and retries
the raw call exactly once; a second failure raises the typed exn.

Retry eligibility: a retried call must be effect-free *on failure*,
which the integer-status contract guarantees for handles and outputs
but NOT for the global RNG stream — ATen ops that draw (randn/rand,
dropout's training path) may consume generator state before the
failing allocation, so a blind retry would advance the stream and
silently break seeded parity. Policy: the retry wrap applies only to
RNG-free bindings — a static property the binding layer knows
(exclude the random.h creation family and dropout; everything else in
the current surface is deterministic). If an RNG-consuming op ever
needs the retry, snapshot/restore of generator state is the mechanism,
as its own considered change.

With leg 1's pressure this path is rare; it exists for the measured
failure mode — an unlucky allocation striking between majors with dead
handles pending — and turns it into transparent recovery instead of a
user-visible error.

The old "no GC-and-retry" non-goal below is narrowed: what stays out of
scope is a C-side callback into Racket mid-allocation; the Racket-side
retry above replaces it.

### Leg 2: observability + good-neighbor knobs

- `tr_cuda_memory_allocated` / `tr_cuda_memory_reserved` (caching-
  allocator stats): lets tests assert per-process VRAM without
  nvidia-smi (whose board totals contaminated the first investigation).
- `tr_cuda_empty_cache`: return reserved-but-unused blocks to the
  driver — scripts call it after training so a finished process isn't
  squatting on reservation high-water.

### Explicit non-goals (for now)

- No C-side GC callback into Racket mid-allocation (the leg-1.5
  Racket-side collect-and-retry covers the recoverable case without
  the re-entrancy).
- No per-storage accounting, no custodian scoping, no allocator
  configuration surface (`PYTORCH_CUDA_ALLOC_CONF` stays an env-var
  note in docs).

## Validation gates (all pre-existing probes)

1. Churn A/B (above): high-water bounded with the wrap in place, no
   change to results.
2. mem-probe at train-novel scale on an idle card: per-process VRAM
   (now via `tr_cuda_memory_allocated`) flat AND ≈ working set.
3. The 22 GiB-balloon training run: with legs 1 + 1.5 it should
   *complete* (collect-and-retry absorbs the squeeze); if genuinely
   unsatisfiable it fails with one typed `exn:fail:rktorch:oom` —
   never the cascade. The balloon probe gains a catch asserting the
   exn type.
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
| C | leg 1.5: tr_last_error_kind + typed oom exn + collect-and-retry | medium, cpp + racket |
| D | cuda memory stats + empty-cache + script/docs wiring | small |

A lands first (kills the crash class); B makes the footprint honest;
C makes OOM catchable and usually invisible; D makes it all verifiable
in-repo. C wants B's choke point but not its semantics — they can land
in either order after A.
