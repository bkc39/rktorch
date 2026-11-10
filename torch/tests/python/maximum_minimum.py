"""torch.maximum and torch.minimum over broadcast, mixed-dtype, NaN and
integer operands.

Each operand is torch.tensor(values).to(dtype).reshape(shape), which the
Racket side mirrors as (reshape (to-dtype (tensor values) dtype) shape ...).
Values travel as Racket-readable strings, because JSON carries neither NaN
nor the infinities through read-json; integers stay exact, past 2**53, and
a bool reads back as rktorch's tensor->list does, 0.0 or 1.0.
"""
import json
import math

import torch

NAN = math.nan
INF = math.inf
BIG = 2**53

CASES = [
    ("broadcast-column-row",
     ([0.0, 2.0], [2, 1], "float32"), ([-1.0, 1.0, 3.0], [3], "float32")),
    ("broadcast-rank-3",
     ([1.5, -2.5, 0.25, 4.0, -1.0, 2.0], [2, 1, 3], "float32"),
     ([0.5, 1.0, -3.0, 2.0], [4, 1], "float32")),
    ("nan-left", ([NAN, 1.0, NAN, -INF], [4], "float32"),
     ([2.0, 3.0, 0.5, 1.0], [4], "float32")),
    ("nan-right", ([2.0, 3.0, 0.5, INF], [4], "float32"),
     ([NAN, 1.0, NAN, -1.0], [4], "float32")),
    ("nan-both", ([NAN, NAN], [2], "float32"), ([NAN, 1.0], [2], "float32")),
    ("nan-broadcast", ([1.0, -2.0, 3.0], [3], "float32"),
     ([NAN], [1], "float32")),
    ("nan-float64-float32", ([NAN, 0.125, -7.5], [3], "float64"),
     ([1.0, NAN, -8.0], [3], "float32")),
    ("int64-float32", ([5, -2, 7, 0], [4], "int64"),
     ([4.5, -3.0, 7.5, NAN], [4], "float32")),
    ("float16-float32", ([1.5, -0.25, 8.0], [3], "float16"),
     ([1.0, 0.5, 9.5], [3], "float32")),
    ("float16-int64", ([1.5, -0.25, NAN], [3], "float16"),
     ([2, -1, 0], [3], "int64")),
    ("bfloat16-float16", ([1.5, -4.0], [2], "bfloat16"),
     ([2.0, -8.0], [2], "float16")),
    ("uint8-int64", ([3, 200, 0], [3], "uint8"), ([-5, 255, 1], [3], "int64")),
    ("bool-float32", ([1, 0, 1], [3], "bool"), ([0.5, -1.0, 2.0], [3], "float32")),
    ("bool-int64", ([1, 0], [2], "bool"), ([-3, 4], [2], "int64")),
    ("int64-int64",
     ([BIG + 1, -(BIG + 1), 9, -4], [2, 2], "int64"),
     ([BIG, -BIG, 11, -4], [2, 2], "int64")),
    ("int64-broadcast", ([3, -6, 10], [3, 1], "int64"),
     ([0, 5], [2], "int64")),
    ("uint8-uint8", ([0, 128, 255], [3], "uint8"), ([1, 127, 254], [3], "uint8")),
    ("bool-bool", ([1, 0, 0, 1], [4], "bool"), ([0, 0, 1, 1], [4], "bool")),
]


def rkt(v):
    if isinstance(v, bool):
        return "1.0" if v else "0.0"
    if isinstance(v, int):
        return str(v)
    if math.isnan(v):
        return "+nan.0"
    if math.isinf(v):
        return "+inf.0" if v > 0 else "-inf.0"
    return repr(v)


def operand(spec):
    values, shape, dtype = spec
    t = torch.tensor(values).to(getattr(torch, dtype)).reshape(shape)
    return t, {"values": [rkt(v) for v in values], "shape": shape,
               "dtype": dtype}


def result(t):
    return {"values": [rkt(v) for v in t.flatten().tolist()],
            "shape": list(t.shape),
            "dtype": str(t.dtype).removeprefix("torch.")}


def case(name, a_spec, b_spec):
    a, a_json = operand(a_spec)
    b, b_json = operand(b_spec)
    return {"name": name, "a": a_json, "b": b_json,
            "maximum": result(torch.maximum(a, b)),
            "minimum": result(torch.minimum(a, b))}


print(json.dumps([case(*c) for c in CASES]))
