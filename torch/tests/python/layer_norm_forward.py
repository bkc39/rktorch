"""nn.LayerNorm forward on a seeded input.

LayerNorm's init is deterministic (ones/zeros), so the forward is the
meaningful parity check for the Racket layer.
"""
import json
import torch
import torch.nn as nn

torch.manual_seed(0)
m = nn.LayerNorm(5)
x = torch.randn(3, 5)
r = m(x)
print(json.dumps({
    "shape": list(r.shape),
    "values": [float(v) for v in r.detach().flatten().tolist()],
}))
