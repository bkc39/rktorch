"""CUDA twin of trainer-precision-test.rkt's '16-mixed fits (G3-L15, #385).

A plain PyTorch loop doing what `fit #:precision '16-mixed
#:gradient-clip-val 1.0` does, as Lightning's MixedPrecision plugin: a
Linear(8, 4) built and fed on the CUDA device under seed 0, 32 random rows in
batches of 8, SGD with momentum, two epochs; the forward and loss under
torch.autocast("cuda", torch.float16), the loss scaled by a GradScaler, the
gradients unscaled before clip_grad_norm_ at 1.0, then the scaler's step and
update. With RKTORCH_AMP_INF_AT set to a step number, that step's scaled loss
is multiplied by inf, so the scaler skips it and halves the scale.

With RKTORCH_AMP_GAP set, gradients accumulate over windows of two batches
and each epoch's batch 1, a window's end, returns None. Lightning 2.6.5's
MixedPrecision.optimizer_step then sets skip_unscaling: it still calls
on_before_optimizer_step and clips the still-scaled gradients, but runs no
unscale_, scaler.step or scaler.update, so the window's batch 0 is dropped.

Prints every computed batch loss and the parameters after training.
"""

import json
import os

import torch
from torch import nn

DEVICE = os.environ.get("RKTORCH_PARITY_DEVICE") or "cuda"
INF_AT = os.environ.get("RKTORCH_AMP_INF_AT")
GAP = os.environ.get("RKTORCH_AMP_GAP") is not None
ACCUMULATE = 2 if GAP else 1

with torch.device(DEVICE):
    torch.manual_seed(0)
    net = nn.Linear(8, 4)
    xs, ys = torch.randn(32, 8), torch.randn(32, 4)
opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
scaler = torch.amp.GradScaler(device="cuda")
losses = []
step = 0
for _ in range(2):
    for i in range(4):
        x, y = xs[8 * i:8 * i + 8], ys[8 * i:8 * i + 8]
        if i % ACCUMULATE == 0:
            opt.zero_grad(set_to_none=True)
        if GAP and i == 1:
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            step += 1
            continue
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.mse_loss(net(x), y)
        losses.append(loss.item())
        factor = float("inf") if INF_AT is not None and step == int(INF_AT) else 1.0
        scaler.scale(loss * factor / ACCUMULATE).backward()
        if (i + 1) % ACCUMULATE == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1

params = torch.cat([p.detach().flatten() for p in net.parameters()])

print(json.dumps({
    "shape": list(params.shape),
    "values": params.tolist(),
    "losses": losses,
    "scale": scaler.get_scale(),
}))
