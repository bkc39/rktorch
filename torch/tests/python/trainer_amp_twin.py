"""CUDA twin of trainer-precision-test.rkt's '16-mixed fit (G3-L15, #385).

A plain PyTorch loop doing what `fit #:precision '16-mixed
#:gradient-clip-val 1.0` does, as Lightning's MixedPrecision plugin: a
Linear(8, 4) built and fed on the CUDA device under seed 0, 32 random rows in
batches of 8, SGD with momentum, two epochs; the forward and loss under
torch.autocast("cuda", torch.float16), the loss scaled by a GradScaler, the
gradients unscaled before clip_grad_norm_ at 1.0, then the scaler's step and
update. Prints every batch's loss and the parameters after training.
"""

import json
import os

import torch
from torch import nn

DEVICE = os.environ.get("RKTORCH_PARITY_DEVICE") or "cuda"

with torch.device(DEVICE):
    torch.manual_seed(0)
    net = nn.Linear(8, 4)
    xs, ys = torch.randn(32, 8), torch.randn(32, 4)
opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
scaler = torch.amp.GradScaler(device="cuda")
losses = []
for _ in range(2):
    for i in range(0, 32, 8):
        x, y = xs[i:i + 8], ys[i:i + 8]
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.mse_loss(net(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())

params = torch.cat([p.detach().flatten() for p in net.parameters()])

print(json.dumps({
    "shape": list(params.shape),
    "values": params.tolist(),
    "losses": losses,
}))
