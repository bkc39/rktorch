"""Numeric twin of trainer-schedule-test.rkt's G2-L12 (T2 spec, #385).

A plain PyTorch loop, no Lightning, doing what `fit` does for the test's DINO
trainer: a seeded Linear(4, 2) on 16 seeded random rows in batches of four,
two epochs, AdamW over two parameter groups (the weight at weight decay
0.04, the bias at 0) with lr 1e-2. A SequentialLR of a three-step linear
warm-up and a cosine annealing steps after every optimizer step, and before
every batch group 0's weight decay is written from the curve of the global
step, as param-group-scheduler does at on-train-batch-start.

Prints the loss of every batch, and the parameters after training.
"""

import json
import math
import os

import torch
from torch import nn

DEVICE = os.environ.get("RKTORCH_PARITY_DEVICE") or "cpu"

S = torch.optim.lr_scheduler


def dino_wd(t):
    return 0.04 + 0.1 * (1.0 - math.cos(math.pi * t / 8))


torch.manual_seed(0)
with torch.device(DEVICE):
    net = nn.Linear(4, 2)
xs = torch.randn(16, 4).to(DEVICE)
ys = torch.randn(16, 2).to(DEVICE)
batches = [(xs[i:i + 4], ys[i:i + 4]) for i in range(0, 16, 4)]

opt = torch.optim.AdamW([{"params": [net.weight], "weight_decay": 0.04},
                         {"params": [net.bias], "weight_decay": 0.0}], lr=1e-2)
sched = S.SequentialLR(
    opt, [S.LinearLR(opt, start_factor=0.1, total_iters=3),
          S.CosineAnnealingLR(opt, T_max=5, eta_min=1e-4)], milestones=[3])

losses = []
step = 0
for _ in range(2):
    for x, y in batches:
        opt.param_groups[0]["weight_decay"] = dino_wd(step)
        loss = nn.functional.mse_loss(net(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        sched.step()
        losses.append(loss.item())

params = torch.cat([p.detach().flatten() for p in net.parameters()])

print(json.dumps({
    "shape": list(params.shape),
    "values": params.tolist(),
    "losses": losses,
}))
