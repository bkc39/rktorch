"""Parity twin of examples/racket/02-matmul.rkt: a @ a.T."""

import json

import torch

a = torch.arange(0.0, 6.0, 1.0).reshape(2, 3)
t = a @ a.T

print(json.dumps({
    "shape": list(t.shape),
    "values": t.flatten().tolist(),
    "repr": repr(t),
}))
