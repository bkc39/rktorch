"""Parity twin of examples/racket/04-mlp.rkt: seeded 4-8-2 MLP, 5 SGD steps;
prints per-step losses + flattened post-training parameters."""

import json

import torch
from torch import nn

torch.manual_seed(0)


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


model = MLP(4, 8, 2)
x = torch.randn(16, 4)
y = torch.randn(16, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.1)

losses = []
for _ in range(5):
    opt.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    opt.step()
    losses.append(loss.item())

params = torch.cat([p.detach().flatten() for p in model.parameters()])

print(json.dumps({
    "shape": list(params.shape),
    "values": params.tolist(),
    "losses": losses,
}))
