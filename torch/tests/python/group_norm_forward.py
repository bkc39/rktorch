"""nn.GroupNorm forward on a seeded input.

GroupNorm's init is deterministic (ones/zeros), so the forward is the
meaningful parity check for the Racket GroupNorm layer.
"""
import json
import torch
import torch.nn as nn

torch.manual_seed(0)
m = nn.GroupNorm(2, 4)
x = torch.randn(2, 4, 3, 3)
r = m(x)
print(json.dumps({
    "shape": list(r.shape),
    "values": [float(v) for v in r.detach().flatten().tolist()],
}))
