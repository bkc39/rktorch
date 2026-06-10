"""Reference MLP training run for examples/racket/04-mlp.rkt.

Same seed as the Racket side: build a 4-8-2 MLP (two nn.Linear layers, so
the seeded init draws match), sample a fixed batch, take 5 SGD steps on MSE
loss. Prints the per-step losses and the flattened post-training parameters
as {"shape": ..., "values": ..., "losses": [...]}.
"""

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
