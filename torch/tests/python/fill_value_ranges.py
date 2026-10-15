"""torch.full at each dtype's boundary: what it fills, or that it refuses.

Values travel as Racket-readable strings, so infinities, NaN and integers
past 2^53 survive the JSON; a bool fill reads back as rktorch's item does,
0.0 or 1.0.
"""
import json
import math
import random

import torch

random.seed(284)


def rkt(v):
    if isinstance(v, int):
        return str(v)
    if math.isnan(v):
        return "+nan.0"
    if math.isinf(v):
        return "+inf.0" if v > 0 else "-inf.0"
    return repr(v)


def steps(x, n, direction):
    out = []
    for _ in range(n):
        x = math.nextafter(x, direction)
        out.append(x)
    return out


def float_probes(dtype):
    top = float(torch.finfo(dtype).max)
    below = torch.nextafter(torch.tensor(top, dtype=dtype),
                            torch.tensor(0.0, dtype=dtype)).item()
    rounding_edge = top + (top - below) / 2
    jitter = (top * random.uniform(0.999, 1.001) for _ in range(16))
    near = [top, below, rounding_edge]
    near += steps(top, 3, 0.0) + steps(top, 3, math.inf)
    near += steps(rounding_edge, 2, 0.0) + steps(rounding_edge, 2, math.inf)
    near += [v for v in jitter if math.isfinite(v)]
    return [s * v for v in near for s in (1, -1)] + [
        math.inf, -math.inf, math.nan, 0.0, -0.0, 1e-50]


def int_probes(lo, hi):
    edges = [lo + k for k in range(-2, 3)] + [hi + k for k in range(-2, 3)]
    return edges + [0.5, -0.5, float(lo), math.inf, -math.inf, math.nan]


PROBES = {
    "float16": float_probes(torch.float16),
    "bfloat16": float_probes(torch.bfloat16),
    "float32": float_probes(torch.float32),
    "float64": float_probes(torch.float64) + [2**1024, -(2**1024)],
    "int64": int_probes(-2**63, 2**63 - 1)
    + [2**53 + k for k in range(-2, 3)] + [-(2**53) - 1],
    "uint8": int_probes(0, 255),
    "bool": int_probes(0, 1) + [2, -1],
}


def fill(v, dtype):
    try:
        out = torch.full((1,), v, dtype=getattr(torch, dtype))
    except (RuntimeError, OverflowError):
        return "refused"
    return rkt(out.float().item() if dtype == "bool" else out.item())


print(json.dumps({d: [[rkt(v), fill(v, d)] for v in vs]
                  for d, vs in PROBES.items()}))
