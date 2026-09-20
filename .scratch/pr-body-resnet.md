Leg 3 of #152: ResNet on CIFAR-10, the classic supervised recipe, on the
three earlier legs. Stacked on #167 (leg 2); the base retargets as the
stack merges.

## What lands

- **`torch/vision/resnet.rkt`**: `BasicBlock` (two 3x3 convolutions under
  batch norm, a projection shortcut only where the shape changes and no
  field otherwise, as torchvision) and `ResNet` (`#:classes #:base
  #:blocks`; a 3x3 stem for 32x32 images, four stages doubling the width,
  global average pooling, a linear head; ResNet-18 at base 64 by default).
  `Conv2d` gains `#:bias?` so the convolutions under batch norm draw no
  bias, matching `bias=False`.
- **`examples/racket/09-resnet.rkt`** with its twin and harness: the MNIST
  loop plus device-side random crop and flip from the loader's generator,
  SGD with Nesterov momentum and weight decay under `one-cycle-lr` stepped
  per batch, and the forward under `with-autocast` on CUDA with the backward
  outside. `run-example` is the seeded float32 core on the fixture (the
  narrow base-16 net, five full-batch steps); `train-cifar10` is the
  headline run reporting test accuracy per epoch.
- Manual section "ResNet" in the vision chapter; AGENTS rosters.

## Verification

- CPU (`.#ci`): the example harness (two steps on eight fixture images,
  the parameter tree pinned, accuracy leaves the net in train mode), nn,
  nn-contract, transforms, convnet-smoke and the MNIST harness; `raco
  review`, `resyntax analyze` against master, scribble.
- GPU (`.#cuda`): python-cross with the `09_resnet.py` training twin
  (losses and all parameters after five steps, CPU at 1e-4 and CUDA at
  5e-3) and the harness's accelerator arm.
- The headline run's per-epoch test accuracy on the 3090 Ti follows in a
  comment.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01VDEpCNkMi2rmxjgRnCHhkp
