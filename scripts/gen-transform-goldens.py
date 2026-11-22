"""Regenerates torch/vision/fixtures/convert-normalize.rktd, torchvision's
own convert_image_dtype and normalize on fixed inputs:

    nix develop --command python3 scripts/gen-transform-goldens.py

Every value is written as the float64 that holds it exactly, so the Racket
test compares bit for bit without Python on CI.
"""
import os

import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.transforms.v2 import functional as F2

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "torch", "vision", "fixtures", "convert-normalize.rktd")

DTYPES = {"uint8": torch.uint8, "float16": torch.float16,
          "bfloat16": torch.bfloat16, "float32": torch.float32,
          "float64": torch.float64}
FLOATS = ["float16", "bfloat16", "float32", "float64"]

BYTES = list(range(256))
BYTES_SHAPE = [1, 16, 16]
UNIT = [0.0, 1.0, 0.5, 0.25, 0.999, 0.001] + [(37 * k) % 101 / 100
                                              for k in range(42)]
UNIT_SHAPE = [3, 4, 4]
BATCH = UNIT + [1.0 - v for v in UNIT]
BATCH_SHAPE = [2, 3, 4, 4]
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def floats(values, shape, dtype):
    return torch.tensor(values, dtype=torch.float64).reshape(shape).to(dtype)


def source(name):
    if name == "uint8":
        return torch.tensor(BYTES, dtype=torch.uint8).reshape(BYTES_SHAPE)
    return floats(UNIT, UNIT_SHAPE, DTYPES[name])


def values(t):
    return t.to(torch.float64).flatten().tolist()


def sexp(v):
    if isinstance(v, bool):
        return "#t" if v else "#f"
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, (int, str)):
        return str(v)
    return "(" + " ".join(sexp(x) for x in v) + ")"


def convert_case(src, target):
    got = F.convert_image_dtype(source(src), DTYPES[target])
    return [src, target, values(got)]


def normalize_case(dtype, inputs, shape, mean, std, inplace):
    x = floats(inputs, shape, DTYPES[dtype])
    y = x.clone()
    want = F.normalize(x.clone(), mean, std, inplace=inplace)
    got = F2.normalize(y, mean, std, inplace=inplace)
    assert torch.equal(want, got) and want.dtype == got.dtype
    assert (got is y) == inplace
    return [dtype, "image" if inputs is UNIT else "batch", mean, std,
            inplace, values(got)]


normalize_cases = [
    [UNIT, UNIT_SHAPE, *IMAGENET, False],
    [BATCH, BATCH_SHAPE, *IMAGENET, True],
    [UNIT, UNIT_SHAPE, [0.5], [0.25], False],
    [UNIT, UNIT_SHAPE, [0.0, 0.0, 0.0], [-2.0, 1.0, 0.5], True],
]

with open(out, "w") as f:
    f.write(f";; scripts/gen-transform-goldens.py, torch {torch.__version__}, "
            f"torchvision {torchvision.__version__}\n")
    f.write("((bytes " + sexp(BYTES_SHAPE) + " " + sexp(BYTES) + ")\n")
    f.write(" (image " + sexp(UNIT_SHAPE) + " " + sexp(UNIT) + ")\n")
    f.write(" (batch " + sexp(BATCH_SHAPE) + " " + sexp(BATCH) + ")\n")
    f.write(" (convert\n")
    for src in DTYPES:
        for target in DTYPES:
            f.write("  " + sexp(convert_case(src, target)) + "\n")
    f.write(" )\n (normalize\n")
    for dtype in FLOATS:
        for case in normalize_cases:
            f.write("  " + sexp(normalize_case(dtype, *case)) + "\n")
    f.write(" ))\n")
print(f"wrote {out}")
