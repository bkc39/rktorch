"""torch.optim's three implementations on the seeded 4-8-2 MLP of 04_mlp.py:
the flattened parameters after five steps with foreach=False, foreach=True and
fused=True, per configuration, on the CPU and on CUDA when present. The Racket
twin is the optim_multi_tensor block of python-cross-test.rkt."""
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


configs = {
    "sgd": (torch.optim.SGD, {"lr": 0.1}, True),
    "sgd_nesterov_wd": (torch.optim.SGD, {"lr": 0.1, "momentum": 0.9,
                                          "nesterov": True,
                                          "weight_decay": 5e-4}, True),
    "adam": (torch.optim.Adam, {"lr": 0.05}, True),
    "adam_wd": (torch.optim.Adam, {"lr": 0.05, "weight_decay": 1e-2}, True),
    "adamw": (torch.optim.AdamW, {"lr": 0.05}, True),
    "rmsprop": (torch.optim.RMSprop, {"lr": 0.01}, False),
    "rmsprop_momentum_wd": (torch.optim.RMSprop, {"lr": 0.01, "momentum": 0.9,
                                                  "weight_decay": 1e-3}, False),
}
modes = {"loop": {"foreach": False}, "foreach": {"foreach": True},
         "fused": {"fused": True}}


def run(cls, kwargs, mode, device):
    torch.manual_seed(0)
    model = MLP().to(device)
    x = torch.randn(16, 4).to(device)
    y = torch.randn(16, 2).to(device)
    opt = cls(model.parameters(), **kwargs, **modes[mode])
    for _ in range(5):
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        opt.step()
    return torch.cat([p.detach().flatten() for p in model.parameters()]).cpu().tolist()


def on(device):
    return {name: {mode: run(cls, kwargs, mode, device)
                   for mode in modes if mode != "fused" or fusable}
            for name, (cls, kwargs, fusable) in configs.items()}


print(json.dumps({"cpu": on("cpu"),
                  "cuda": on("cuda") if torch.cuda.is_available() else None}))
