"""Reference computation for examples/racket/01-arith.rkt.

Deterministic (no RNG): (x + 1) * relu(x) over a fixed 2x2 tensor, printed as
{"shape": [...], "values": [...], "repr": "..."} for the parity cross-test.
"""

import json

import torch

x = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
t = (x + 1) * x.relu()

print(json.dumps({
    "shape": list(t.shape),
    "values": t.flatten().tolist(),
    "repr": repr(t),
}))
