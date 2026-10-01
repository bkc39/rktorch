"""F.scaled_dot_product_attention: values and gradients at dropout_p = 0.

The twin of torch/tests/attention-parity-test.rkt. Every tensor is drawn
seeded on the CPU, in the Racket test's order, then moved to
RKTORCH_PARITY_DEVICE, so the CUDA leg runs libtorch's fused kernels on
the same numbers. The loss weighs each output by a drawn tensor, so the
gradients are not the plain sums a bare `sum()` would give.
"""
import json
import os

import torch
import torch.nn.functional as F

device = os.environ.get("RKTORCH_PARITY_DEVICE", "cpu")


def flat(t):
    return [float(v) for v in t.detach().cpu().flatten().tolist()]


def leaf(*shape):
    return torch.randn(*shape).to(device).requires_grad_()


def case(length, make_mask=lambda: None, **options):
    q = leaf(2, 2, length, 4)
    k = leaf(2, 2, 5, 4)
    v = leaf(2, 2, 5, 6)
    mask = make_mask()
    if mask is not None:
        mask = mask.to(device)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, **options)
    weights = torch.randn(*out.shape).to(device)
    (out * weights).sum().backward()
    return {
        "shape": list(out.shape),
        "out": flat(out),
        "grads": [flat(t.grad) for t in (q, k, v)],
    }


def padding():
    rows = [[True] * 5, [True] * 3 + [False] * 2]
    return torch.tensor(rows).reshape(2, 1, 1, 5)


torch.manual_seed(0)
print(json.dumps({
    "plain": case(3),
    "bool_mask": case(3, padding),
    "float_mask": case(3, lambda: torch.randn(3, 5)),
    "causal": case(5, is_causal=True),
    "scale": case(3, scale=0.3),
}))
