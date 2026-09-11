"""Build on CPU, move with .to(DEVICE), then train: the load-then-move flow.

Adam is constructed BEFORE the move on purpose: PyTorch creates its moments
lazily at the first step, on the parameter's device, and the Racket twin
must do the same. DEVICE comes from RKTORCH_PARITY_DEVICE (default cpu).
"""
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = os.environ.get("RKTORCH_PARITY_DEVICE") or "cpu"

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
xs = torch.randn(16, 4)
ys = xs @ torch.ones(4, 1)
opt = torch.optim.Adam(model.parameters(), lr=0.1)

model.to(DEVICE)
xs, ys = xs.to(DEVICE), ys.to(DEVICE)

losses = []
for _ in range(3):
    opt.zero_grad()
    loss = F.mse_loss(model(xs), ys)
    loss.backward()
    opt.step()
    losses.append(float(loss))

model.to("cpu")
flat = torch.cat([p.detach().flatten() for p in model.parameters()])
print(json.dumps({
    "losses": losses,
    "params": [float(v) for v in flat.tolist()],
    "device": DEVICE,
}))
