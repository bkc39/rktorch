"""Reference computation for examples/racket/02-matmul.rkt.

Deterministic: the Gram matrix a @ a.T of a = arange(6).reshape(2, 3), printed
as {"shape": [...], "values": [...], "repr": "..."} for the parity cross-test.
"""

import json

import torch

a = torch.arange(0.0, 6.0, 1.0).reshape(2, 3)
t = a @ a.T

print(json.dumps({
    "shape": list(t.shape),
    "values": t.flatten().tolist(),
    "repr": repr(t),
}))
