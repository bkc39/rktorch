"""Draws from a seeded generator on RKTORCH_PARITY_DEVICE.

The twin of torch/tests/generator-device-test.rkt (R2): three draws in the
Racket test's order from one torch.Generator(device).manual_seed(7).
"""
import json
import os

import torch

device = os.environ.get("RKTORCH_PARITY_DEVICE", "cpu")


def flat(t):
    return [float(v) for v in t.cpu().flatten().tolist()]


g = torch.Generator(device).manual_seed(7)
print(json.dumps({
    "rand": flat(torch.rand(5, device=device, generator=g)),
    "randn": flat(torch.randn(3, 2, device=device, generator=g)),
    "rand64": flat(torch.rand(4, device=device, dtype=torch.float64,
                              generator=g)),
}))
