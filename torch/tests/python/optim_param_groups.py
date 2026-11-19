"""torch.optim's parameter groups on the seeded 4-8-2 MLP of 04_mlp.py: the
flattened parameters after five steps, the weights in the first group and the
biases in the second, with foreach=False, foreach=True and fused=True, and
SGD with momentum over one group whose weight_decay is written after step 2.
CPU only. The Racket twin is optim-param-groups-parity-test.rkt."""
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
    "adamw": (torch.optim.AdamW, {"lr": 0.05},
              [{"weight_decay": 0.5}, {"weight_decay": 0.0, "lr": 2e-3}], True),
    "sgd_momentum": (torch.optim.SGD, {"lr": 0.1},
                     [{"momentum": 0.9}, {"momentum": 0.5}], True),
    "rmsprop_alpha": (torch.optim.RMSprop, {"lr": 0.01},
                      [{"alpha": 0.9}, {"alpha": 0.99}], False),
}
modes = {"loop": {"foreach": False}, "foreach": {"foreach": True},
         "fused": {"fused": True}}


def run(make_opt, after=lambda opt, n: None):
    torch.manual_seed(0)
    model = MLP()
    x = torch.randn(16, 4)
    y = torch.randn(16, 2)
    decay = [model.fc1.weight, model.fc2.weight]
    no_decay = [model.fc1.bias, model.fc2.bias]
    opt = make_opt(decay, no_decay)
    for n in range(1, 6):
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        opt.step()
        after(opt, n)
    return torch.cat([p.detach().flatten() for p in model.parameters()]).tolist()


def grouped(cls, kwargs, group_kwargs, mode):
    def make(decay, no_decay):
        groups = [{"params": ps, **kw} for ps, kw in zip([decay, no_decay], group_kwargs)]
        return cls(groups, **kwargs, **modes[mode])
    return make


def set_weight_decay(opt, n):
    if n == 2:
        opt.param_groups[0]["weight_decay"] = 0.01


print(json.dumps({
    "groups": {name: {mode: run(grouped(cls, kwargs, group_kwargs, mode))
                      for mode in modes if mode != "fused" or fusable}
               for name, (cls, kwargs, group_kwargs, fusable) in configs.items()},
    "set_weight_decay": run(lambda decay, no_decay:
                            torch.optim.SGD([{"params": decay + no_decay}], lr=0.1,
                                            momentum=0.9),
                            set_weight_decay),
}))
