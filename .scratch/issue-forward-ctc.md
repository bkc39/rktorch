#152 leg 0 added `[id : contract]` to `define-layer`'s `#:forward` formals, so a
layer states the shape it accepts instead of checking it with an `unless` guard
(AGENTS.md: "a guard that only inspects an argument's shape belongs in the
signature"). `BatchNorm2d` and `BatchNorm1d` use it. Three things were left
alone deliberately, to keep that PR narrow.

## What is left

- **`UNet` still raises its own error.** `torch/vision/diffusion.rkt`'s forward
  hand-checks `[N 3 32 32]`, one int64 timestep per image, and the label
  argument against the network's class count, then calls
  `raise-arguments-error`. The rank and dtype parts belong in the formals; the
  cross-argument part (labels iff the net is conditional, one per image) needs a
  dependent contract, so this is the case that shows whether the clause wants an
  `->i`-shaped form as well as a per-formal one.
- **`Upsample` blames the wrong module.** It leans on
  `upsample-nearest2d`'s `image-batch/c`, whose boundary is between
  `torch/foreign` and `torch/vision/diffusion.rkt`, so a bad rank from a user
  blames diffusion.rkt. Declaring the formal fixes the message. The same holds
  for every layer that currently relies on a promoted op's contract firing.
- **#99's half.** That issue proposes the same `[id : contract]` notation for
  `#:init` formals, which is still open; the forward half now exists, so the
  two should end up spelled the same way.

## Also worth deciding

The party blamed for a forward violation is the label `caller`, not the
importing module: the wrapper is built where the layer is defined, and a
forward has no module boundary of its own to name the caller by. Naming the
real module would mean contracting the layer *value* at the constructor's
result, which chaperones every application. Worth measuring before choosing, on
the cost table's terms.

Refs: #96 (validation via contracts), #99 (`#:init` notation), #117
(define-layer syntax), #152.
