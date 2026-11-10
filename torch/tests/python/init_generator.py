"""Draws from one torch.Generator(device) per case, on RKTORCH_PARITY_DEVICE.

The twin of torch/tests/init-generator-parity-test.rkt (G9): each case seeds
its own generator, draws in the Racket test's order, then takes rand(3) from
it, so a draw that uses the stream differently shows up in "after".
"""
import json
import math
import os

import torch
from torch.nn import init

device = os.environ.get("RKTORCH_PARITY_DEVICE", "cpu")


def flat(t):
    return [float(v) for v in t.cpu().flatten().tolist()]


def empty(*dims, dtype=torch.float32):
    return torch.empty(*dims, device=device, dtype=dtype)


def case(seed, draw):
    g = torch.Generator(device).manual_seed(seed)
    drawn = draw(g)
    return {"drawn": [flat(t) for t in drawn],
            "after": flat(torch.rand(3, device=device, generator=g))}


bound = 1 / math.sqrt(4)
print(json.dumps({
    "uniform": case(1, lambda g: [empty(4, 3).uniform_(-0.5, 0.5, generator=g)]),
    "uniform-defaults": case(2, lambda g: [empty(5).uniform_(generator=g)]),
    "uniform-half": case(3, lambda g: [empty(6, dtype=torch.float16).uniform_(0, 65504, generator=g)]),
    "uniform-point": case(4, lambda g: [empty(5).uniform_(0.25, 0.25, generator=g)]),
    "normal": case(5, lambda g: [empty(7).normal_(0.5, 2.0, generator=g)]),
    "normal-double": case(6, lambda g: [empty(5, dtype=torch.float64).normal_(generator=g)]),
    "normal-zero-std": case(7, lambda g: [empty(5).normal_(1.0, 0.0, generator=g)]),
    "uniform-init": case(8, lambda g: [init.uniform_(empty(2, 3), -1.0, 2.0, generator=g)]),
    "kaiming": case(9, lambda g: [init.kaiming_uniform_(empty(5, 4), a=math.sqrt(5), generator=g)]),
    "kaiming-a0": case(10, lambda g: [init.kaiming_uniform_(empty(3, 2, 3, 3), a=0.0, generator=g)]),
    "normal-init": case(11, lambda g: [init.normal_(empty(3, 4), generator=g)]),
    "trunc": case(12, lambda g: [init.trunc_normal_(empty(64, 32), std=0.02, generator=g)]),
    "trunc-tail": case(13, lambda g: [init.trunc_normal_(empty(64, 32), std=0.02, a=0.03, b=0.05, generator=g)]),
    "linear": case(14, lambda g: [init.kaiming_uniform_(empty(3, 4), a=math.sqrt(5), generator=g),
                                   init.uniform_(empty(3), -bound, bound, generator=g)]),
}))
