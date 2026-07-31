"""nn.Conv2d seeded init: weight + bias values and shapes.

The Racket Conv2d layer's kaiming-uniform weight + uniform bias init
(in that order) must match nn.Conv2d.reset_parameters value-for-value.
"""
import json
import torch
import torch.nn as nn

torch.manual_seed(0)
m = nn.Conv2d(1, 8, 3)
vals = [float(v) for v in m.weight.detach().flatten().tolist()] + [
    float(v) for v in m.bias.detach().flatten().tolist()
]
print(json.dumps({"values": vals,
                  "shapes": [list(m.weight.shape), list(m.bias.shape)]}))
