"""Reference computation for examples/racket/03-autograd.rkt.

Deterministic: d(sum(x*x))/dx == 2x, printed as {"shape": [...], "values":
[...], "repr": "..."} for the parity cross-test.
"""

import json

import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
(x * x).sum().backward()
t = x.grad

print(json.dumps({
    "shape": list(t.shape),
    "values": t.flatten().tolist(),
    "repr": repr(t),
}))
