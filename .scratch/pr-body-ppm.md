Closes #155: sample grids as image files without a dependency, pulled
forward from #84 leg 4 so the DCGAN and VAE capstones of #152 can write
their grids, and the diffusion sampler reuses it.

## What lands

- `torch/vision/ppm.rkt`: `image-grid #:columns #:padding #:pad-value`
  lays an `[N C H W]` batch out as one `[C H' W']` image on the batch's own
  device with torchvision's `make_grid` layout (one channel becomes three);
  `write-ppm path image #:range` writes a `[3 H W]` tensor as binary P6,
  quantizing a float image the way `save_image` does with `#:range` naming
  the values that map to 0 and 255, and writing a uint8 image as it is.
- Tests: the grid layout cell for cell, the header and byte order, the
  half-rounding and clamping, uint8 pass-through, the contracts, device
  tensors; a Python twin (`make_grid_parity.py`) pins the grid against
  `torchvision.utils.make_grid` and the bytes against `save_image`'s
  quantization. The twin falls back to `make_grid`'s own algorithm where the
  Python has no torchvision (the CUDA parity shell carries torch-bin alone).
- Manual section "Images" in the vision chapter; AGENTS.md roster line.

Independent of #158; whichever merges second rebases the vision manual and
the AGENTS.md vision bullets.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01VDEpCNkMi2rmxjgRnCHhkp
