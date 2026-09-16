"""nn.ConvTranspose2d seeded init: weight + bias values and shapes.

The Racket ConvTranspose2d layer's kaiming-uniform weight in the
(in, out, kH, kW) layout, then its uniform bias, must match
nn.ConvTranspose2d.reset_parameters value-for-value.
"""
import json
import torch
import torch.nn as nn

torch.manual_seed(0)
m = nn.ConvTranspose2d(2, 4, 3)
vals = [float(v) for v in m.weight.detach().flatten().tolist()] + [
    float(v) for v in m.bias.detach().flatten().tolist()
]
print(json.dumps({"values": vals,
                  "shapes": [list(m.weight.shape), list(m.bias.shape)]}))
