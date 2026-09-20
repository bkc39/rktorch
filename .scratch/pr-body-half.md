Leg 1 of #152, half precision: the float16 and bfloat16 dtypes end to end,
raw element bytes for the safetensors container, and autocast. Stacked on
#158 (leg 0), which it needs for tranche 6; the base retargets to master
when #158 merges.

## What lands

- **C API**: `TR_DTYPE_FLOAT16` and `TR_DTYPE_BFLOAT16` appended to the
  enum (earlier values stay ABI), the two mappings extended, every
  constructor and conversion inheriting them and values reading back
  through the float32 copy path. `tr_tensor_copy_bytes` / `tr_from_bytes`
  move a tensor's element bytes in its own dtype. `tr_set_autocast_enabled`
  / `tr_is_autocast_enabled` / `tr_autocast_dtype` wrap `at::autocast` per
  device type in the shape of the grad-mode pair; disabling drops the cast
  cache as `torch.autocast`'s exit does. Gtests for the casts and their
  readback, the byte round trip for all seven dtypes with IEEE bit patterns
  pinned, and CPU bfloat16 autocast casting a matmul until disabled.
- **foreign**: the dtype tables mirror the C side; `dtype/c`, the random
  constructors and `tensor`'s `#:dtype` take the half pair (`tensor` builds
  float32 and narrows natively, Racket having no 16-bit float vector); the
  repr names them as PyTorch does; a layer moves to them with its integer
  buffers staying put. `tensor->bytes` and `bytes->tensor`. `with-autocast`,
  `call-with-autocast`, `autocast-enabled?`, `autocast-dtype`: per thread and
  per device type, bfloat16 unless asked for float16, the previous state
  restored by `dynamic-wind` so the form nests and an escape restores.
- **nn**: the safetensors container writes the payload as the element bytes
  with the format's tags for all seven dtypes (F16, BF16, U8 and F64 join
  F32, I64 and BOOL; the existing three are byte-identical to before), reads
  them back through `bytes->tensor` checking the header's shape against the
  model's, and `copy!` converts, so a file in one dtype loads into a model
  in another. `Linear` now runs on a generated `linear` (tranche 6): composed
  from matmul and add, the float32 bias promoted a bf16 product back to
  float32 under autocast, where PyTorch's fused op casts the whole map.
- **Docs**: a "Half precision" section in the devices chapter with the
  training recipe (parameters float32, forward under `with-autocast`,
  `backward!` outside), `with-autocast` and friends, `tensor->bytes` and
  `bytes->tensor`; AGENTS.md rosters.

## Verification

- `nix run .#codegen` leaves a clean tree; `nix build .#cpp .#cpp-format
  .#cpp-tidy` green (three new gtest cases in `dtype_test.cpp`, three in
  `autocast_test.cpp`, one in `generated_tranche6_test.cpp`).
- CPU (`.#ci`): tensor-ops, to, nn, autocast, define-layer, foreign,
  device, selection, bytes-ingestion, diffusion, transforms, nn-contract,
  convnet-smoke, procedure-layer and the MLP and MNIST example harnesses;
  the two tests that asserted float16 is refused now assert it works, and
  two that asserted a float64 buffer cannot be saved now round-trip it.
  `raco review` on the touched files (pre-existing warnings aside),
  `resyntax analyze` against master with no suggestions, scribble builds.
- GPU (`.#cuda`): python-cross with three new twins (`half_dtypes.py`:
  casts, reprs, constructors, a layer moved to bfloat16; `autocast_cpu.py`:
  a matmul and a Linear trained under the cast with float32 grads;
  `safetensors_dtypes.py`: a hand-built file with every tag that Racket
  reads, and whose payloads Racket's writer reproduces byte for byte), the
  autocast suite's CUDA convolution case, generated-parity with the linear
  recipe, and the diffusion suites on the fused Linear.

Throughput and memory of the DDPM UNet under bfloat16 autocast on the 3090
Ti follow in a comment, for #145.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01VDEpCNkMi2rmxjgRnCHhkp
