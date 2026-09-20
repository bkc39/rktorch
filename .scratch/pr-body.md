Leg 0 of #152, the classic vision arc: the op tranche, the layers and the
augmentation the ResNet, DCGAN and VAE capstones need.

## What lands

- **codegen tranche 6**: `batch_norm`, `leaky_relu`,
  `binary_cross_entropy_with_logits`, `huber_loss`, `l1_loss`, `flip`. Every
  one fits the existing kinds, so the generator itself is untouched (the
  sequence arc owns it for #154). Goldens in `generated_tranche6_test.cpp`,
  C pins, parity recipes with training, weighted, sum, none and multi-dim
  drives.
- **foreign**: `batch-norm` (statistics, affine pair, `#:training?`,
  `#:momentum`, `#:eps`; a `->i` precondition insists the statistics come as
  a pair and are present unless training, ATen's rule as contract blame),
  `leaky-relu #:negative-slope`, `flip` with one dim or a list.
- **nn**: `BatchNorm2d` and `BatchNorm1d`, the first `Buffer` customers:
  `running-mean`, `running-var` and an int64 `num-batches-tracked` with
  PyTorch's init, updated by ATen in place in `'train` mode and read in
  `'eval`; state dicts carry them, so a trained BatchNorm round-trips through
  safetensors. `binary-cross-entropy-with-logits`, `huber-loss`, `l1-loss`
  with the mean reduction of their siblings.
- **vision**: `torch/vision/transforms.rkt` with `random-crop #:padding` and
  `random-horizontal-flip #:p`, on the device from ops the library already
  has; one `draw-seed` from the torch generator seeds a Racket generator per
  batch, so a seeded loader replays its augmentation. The draws are the
  transforms' own, not torchvision's.
- Manual section for the transforms; AGENTS.md rosters.

## Verification

- `nix run .#codegen` leaves a clean tree; `nix build .#cpp .#cpp-format
  .#cpp-tidy` green (six new gtests).
- CPU (`.#ci`): nn, nn-contract, tensor-ops, transforms, define-layer,
  foreign, to suites; `raco review` on the touched files (the remaining
  warnings are the pre-existing ctc-loss parens and require order);
  `resyntax analyze` against master: no suggestions; scribble builds.
- GPU (`.#cuda`): python-cross (new `batch_norm_forward.py` twin: training
  forward, running statistics, counter, eval forward) and generated-parity
  with the tranche 6 recipes and drives.

Next legs per #152: half precision, the optimizer surface with schedulers
(#144), `09-resnet.rkt`, then DCGAN and VAE after #155.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01VDEpCNkMi2rmxjgRnCHhkp
