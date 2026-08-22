"""Parity twin of examples/racket/03-autograd.rkt: d(sum(x*x))/dx == 2x."""

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
