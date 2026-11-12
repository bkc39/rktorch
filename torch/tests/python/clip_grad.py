"""torch.nn.utils.clip_grad_norm_ and clip_grad_value_ on the G0-L1 cases of
clip-grad-test.rkt: per case the gradients in and out and, for the norm, the
total. The Racket twin is clip-grad-parity-test.rkt. An infinite norm type
travels as a Racket-readable string: JSON has no infinities."""
import json
import math

import torch

GRADS = [
    ([0.5, -1.25, 2.0], "float32"),
    ([3.0, -0.75], "float64"),
    ([-2.5, 0.125, 1.5, -0.25], "float32"),
]

SINGLE = [([0.75, -0.5, 1.25, 0.0625, -2.0], "float32")]

NORM_TYPES = [2.0, 1.0, 0.5, math.inf]


def rkt(v):
    if math.isinf(v):
        return "+inf.0" if v > 0 else "-inf.0"
    return v


def parameters(grads):
    ps = []
    for values, dtype in grads:
        p = torch.zeros(len(values), dtype=getattr(torch, dtype),
                        requires_grad=True)
        (p * torch.tensor(values, dtype=getattr(torch, dtype))).sum().backward()
        ps.append(p)
    return ps


def flat(ps):
    return [[float(v) for v in p.grad.tolist()] for p in ps]


def norm_case(name, grads, max_norm, norm_type, foreach):
    ps = parameters(grads)
    total = torch.nn.utils.clip_grad_norm_(ps, max_norm, norm_type=norm_type,
                                           foreach=foreach)
    return {"name": name, "grads_in": [g for g, _ in grads],
            "dtypes": [d for _, d in grads], "max_norm": max_norm,
            "norm_type": rkt(norm_type), "foreach": foreach,
            "norm": float(total), "grads_out": flat(ps)}


def value_case(name, grads, clip_value, foreach):
    ps = parameters(grads)
    torch.nn.utils.clip_grad_value_(ps, clip_value, foreach=foreach)
    return {"name": name, "grads_in": [g for g, _ in grads],
            "dtypes": [d for _, d in grads], "clip_value": clip_value,
            "foreach": foreach, "grads_out": flat(ps)}


norms = []
values = []
for foreach in (True, False):
    mode = "foreach" if foreach else "loop"
    for norm_type in NORM_TYPES:
        for label, grads in (("mixed", GRADS), ("single", SINGLE)):
            for max_norm in (1.0, 1e6, 0.0):
                norms.append(norm_case(
                    f"{label} p={norm_type} max={max_norm} {mode}",
                    grads, max_norm, norm_type, foreach))
    norms.append(norm_case(f"empty {mode}", [], 1.0, 2.0, foreach))
    for clip_value in (0.25, 0.1):
        for label, grads in (("mixed", GRADS), ("single", SINGLE)):
            values.append(value_case(f"{label} v={clip_value} {mode}",
                                     grads, clip_value, foreach))

print(json.dumps({"norm": norms, "value": values}))
