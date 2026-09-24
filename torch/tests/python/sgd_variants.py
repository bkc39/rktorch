"""SGD with momentum, Nesterov momentum, and weight decay, and RMSprop with
and without momentum, on the seeded 4-8-2 MLP of 04_mlp.py: per-step losses
and the flattened parameters after five steps, one run per configuration.
"""
import json
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def run(make_opt):
    torch.manual_seed(0)
    model = MLP()
    x = torch.randn(16, 4)
    y = torch.randn(16, 2)
    opt = make_opt(model.parameters())
    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    params = torch.cat([p.detach().flatten() for p in model.parameters()])
    return {"losses": losses, "params": params.tolist()}


configs = {
    "momentum": lambda ps: torch.optim.SGD(ps, lr=0.1, momentum=0.9),
    "nesterov": lambda ps: torch.optim.SGD(ps, lr=0.1, momentum=0.9,
                                           nesterov=True),
    "weight_decay": lambda ps: torch.optim.SGD(ps, lr=0.1, momentum=0.9,
                                               weight_decay=5e-4),
    "adam_weight_decay": lambda ps: torch.optim.Adam(ps, lr=0.05,
                                                     weight_decay=1e-2),
    "rmsprop": lambda ps: torch.optim.RMSprop(ps, lr=0.01),
    "rmsprop_momentum": lambda ps: torch.optim.RMSprop(ps, lr=0.01,
                                                       momentum=0.9,
                                                       weight_decay=1e-3),
}
print(json.dumps({name: run(make) for name, make in configs.items()}))
