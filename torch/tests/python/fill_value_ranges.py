"""torch.full at each dtype's range edges: what it fills, or that it refuses.

Each probe records the fill, "refused", or the documented deviation the
fill shows: "truncates" a fractional integer fill, "wraps" an out-of-range
uint8, "any-real-bool" a bool from anything but 0 or 1, "rounds" a finite
half-precision value past the largest finite one. Values travel as
Racket-readable strings, because JSON carries neither NaN, the infinities
nor -0.0 through read-json; a bool fill reads back as rktorch's item does,
0.0 or 1.0.
"""
import json
import math

import torch


def rkt(v):
    if isinstance(v, int):
        return str(v)
    if math.isnan(v):
        return "+nan.0"
    if math.isinf(v):
        return "+inf.0" if v > 0 else "-inf.0"
    return repr(v)


def float_probes(dtype):
    top = float(torch.finfo(dtype).max)
    below = torch.nextafter(torch.tensor(top, dtype=dtype),
                            torch.tensor(0.0, dtype=dtype)).item()
    rounding_edge = top + (top - below) / 2
    near = [top, below, math.nextafter(top, math.inf), rounding_edge,
            math.nextafter(rounding_edge, 0.0),
            math.nextafter(rounding_edge, math.inf)]
    return [s * v for v in near for s in (1, -1)] + [
        math.inf, -math.inf, math.nan, 0.0, -0.0, 1e-50]


def int_probes(lo, hi):
    edges = [lo - 1, lo, lo + 1, hi - 1, hi, hi + 1]
    return edges + [0.5, -0.5, float(lo), math.inf, -math.inf, math.nan]


FLOATS = ("float16", "bfloat16", "float32", "float64")

PROBES = {
    **{d: float_probes(getattr(torch, d)) for d in FLOATS},
    "int64": int_probes(-2**63, 2**63 - 1) + [2**53 + 1, -(2**53) - 1],
    "uint8": int_probes(0, 255) + [-2],
    "bool": int_probes(0, 1),
}
PROBES["float64"] += [2**1024, -(2**1024)]


def deviation(v, out, dtype):
    if dtype in FLOATS:
        past = math.isfinite(v) and abs(v) > torch.finfo(
            getattr(torch, dtype)).max
        return "rounds" if past else None
    if out == v:
        return None
    if dtype == "bool":
        return "any-real-bool"
    return "wraps" if v == int(v) else "truncates"


def fill(v, dtype):
    try:
        t = torch.full((1,), v, dtype=getattr(torch, dtype))
    except (RuntimeError, OverflowError):
        return {"value": rkt(v), "refused": True}
    out = t.float().item() if dtype == "bool" else t.item()
    return {"value": rkt(v), "fills": rkt(out),
            "deviation": deviation(v, out, dtype)}


print(json.dumps({
    "max": {d: rkt(float(torch.finfo(getattr(torch, d)).max))
            for d in FLOATS},
    "probes": {d: [fill(v, d) for v in vs] for d, vs in PROBES.items()},
}))
