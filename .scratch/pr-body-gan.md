Leg 4 of #152, the last: a DCGAN and a VAE on MNIST, each writing a 10x10
sample grid per epoch through #155's writer. Stacked on #170 (leg 3) with
#159 (the PPM writer) merged in, since the grids need it; the base
retargets as the stack merges, and the merge commit drops out once #159
lands on master.

## What lands

- **`examples/racket/10-dcgan.rkt`** with twin and harness: the DCGAN shrunk
  to 28x28 (a generator of one linear layer and two `ConvTranspose2d` under
  batch norm with a tanh output; a discriminator of two strided `Conv2d`
  under `leaky-relu` and batch norm reading one logit), the standard
  alternating step with `binary-cross-entropy-with-logits` and two `adam`s
  at 2e-4 with `#:beta1 0.5`; the latent drawn on the CPU so a seeded run
  replays on the GPU as the twin draws it.
- **`examples/racket/11-vae.rkt`** with twin and harness: the linear VAE of
  the PyTorch examples, 784-400-20; the forward takes the reparameterization
  noise from the caller and returns three values, the loss is the
  reference's reconstruction cross-entropy plus KL divided by the batch, and
  `decode` maps latents to logits for sampling.
- Both headline runs write `dcgan-epoch-N.ppm` / `vae-epoch-N.ppm` under
  `OUT` from one fixed latent draw, via `image-grid` and `write-ppm`.
- AGENTS roster; harness timeouts.

## Verification

- CPU (`.#ci`): both harnesses (parameter trees, forward shapes, the tanh
  range, a decreasing VAE loss), the ResNet harness, ppm, transforms and nn
  suites; `raco review`, `resyntax analyze` against master, scribble.
- GPU (`.#cuda`): python-cross with `10_dcgan.py` (three alternating steps,
  the interleaved discriminator and generator losses and both networks'
  parameters) and `11_vae.py` (five steps), CPU at 1e-4 and CUDA at 5e-3,
  plus both harnesses' accelerator arms.
- Sample grids from the headline runs follow in a comment.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01VDEpCNkMi2rmxjgRnCHhkp
