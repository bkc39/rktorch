"""torch.flatten from dim 1.

The Racket flatten is reshape logic, not a generated binding, so it
sits outside the manifest battery and gets a direct check.
"""
import json
import torch

torch.manual_seed(0)
x = torch.randn(2, 3, 4)
r = torch.flatten(x, 1)
print(json.dumps({
    "shape": list(r.shape),
    "values": [float(v) for v in r.flatten().tolist()],
}))
