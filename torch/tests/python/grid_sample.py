"""F.affine_grid, F.grid_sample and bicubic F.interpolate: values and gradients.

The twin of torch/tests/grid-sample-parity-test.rkt. Every tensor is drawn
seeded on the CPU, in the Racket test's order, then moved to
RKTORCH_PARITY_DEVICE. The loss weighs each output by a drawn tensor, so the
gradients are not the plain sums a bare `sum()` would give.
"""
import json
import os

import torch
import torch.nn.functional as F

device = os.environ.get("RKTORCH_PARITY_DEVICE", "cpu")


def flat(t):
    return [float(v) for v in t.detach().cpu().flatten().tolist()]


def leaf(t):
    return t.to(device).requires_grad_()


def weighed(out):
    w = torch.randn(*out.shape).to(device)
    (out * w).sum().backward()


def affine(align):
    theta = leaf(torch.randn(2, 2, 3))
    out = F.affine_grid(theta, (2, 3, 4, 5), align_corners=align)
    weighed(out)
    return {"out": flat(out), "grads": [flat(theta.grad)]}


def sampled(mode, padding, align):
    x = leaf(torch.randn(2, 3, 4, 5))
    grid = leaf(torch.rand(2, 6, 7, 2) * 2.4 - 1.2)
    out = F.grid_sample(x, grid, mode=mode, padding_mode=padding,
                        align_corners=align)
    weighed(out)
    return {"out": flat(out), "grads": [flat(x.grad), flat(grid.grad)]}


def resized(align):
    x = leaf(torch.randn(1, 2, 5, 6))
    out = F.interpolate(x, size=(7, 9), mode="bicubic", align_corners=align)
    weighed(out)
    return {"out": flat(out), "grads": [flat(x.grad)]}


torch.manual_seed(0)
cases = {}
for align in (False, True):
    cases[f"affine-{align}"] = affine(align)
for mode in ("bilinear", "nearest", "bicubic"):
    for padding in ("zeros", "border", "reflection"):
        for align in (False, True):
            cases[f"sample-{mode}-{padding}-{align}"] = sampled(
                mode, padding, align)
for align in (False, True):
    cases[f"bicubic-{align}"] = resized(align)
print(json.dumps(cases))
