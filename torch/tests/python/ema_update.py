"""AveragedModel under the EMA rule over three SGD steps on a Linear: the
first update copies the parameters, the later ones decay toward them."""
import json
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

torch.manual_seed(0)
m = nn.Linear(4, 3)
x = torch.randn(8, 4)
avg = AveragedModel(m, multi_avg_fn=get_ema_multi_avg_fn(0.9))
opt = torch.optim.SGD(m.parameters(), lr=0.1)
for _ in range(3):
    opt.zero_grad()
    m(x).pow(2).mean().backward()
    opt.step()
    avg.update_parameters(m)

print(json.dumps({
    "model": torch.cat([p.detach().flatten() for p in m.parameters()]).tolist(),
    "average": torch.cat([p.detach().flatten()
                          for p in avg.module.parameters()]).tolist(),
}))
