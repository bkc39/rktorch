"""Parity twin of examples/racket/00-randn.rkt: seeded randn(2, 2)."""

import json

import torch

torch.manual_seed(0)
t = torch.randn(2, 2)

print(json.dumps({
    "shape": list(t.shape),
    "values": t.flatten().tolist(),
    "repr": repr(t),
}))
