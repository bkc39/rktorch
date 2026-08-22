"""Parity twin of examples/racket/01-arith.rkt: (x + 1) * relu(x)."""

import json

import torch

x = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
t = (x + 1) * x.relu()

print(json.dumps({
    "shape": list(t.shape),
    "values": t.flatten().tolist(),
    "repr": repr(t),
}))
