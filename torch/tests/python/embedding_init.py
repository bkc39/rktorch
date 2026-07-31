"""nn.Embedding seeded init: standard-normal weight values.

The Racket Embedding layer's normal-init (randn) must match
nn.Embedding.reset_parameters (init.normal_) draw for draw.
"""
import json
import torch
import torch.nn as nn

torch.manual_seed(0)
m = nn.Embedding(7, 4)
print(json.dumps({
    "values": [float(v) for v in m.weight.detach().flatten().tolist()],
    "shape": list(m.weight.shape),
}))
