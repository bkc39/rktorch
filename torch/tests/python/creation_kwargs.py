"""The creation kwargs: dtype/device at construction, requires_grad, *_like.

Seeded draws with an explicit dtype must come from the same generator state
as the Racket twin's, so the draw order below is mirrored exactly.
"""
import json
import torch

torch.manual_seed(0)
n64 = torch.randn(2, 2, dtype=torch.float64)
torch.manual_seed(0)
n32 = torch.randn(2, 2)
r64 = torch.rand(3, dtype=torch.float64)
x = torch.zeros(2, 2)
print(json.dumps({
    "randn64": n64.flatten().tolist(),
    "randn32": n32.flatten().tolist(),
    "rand64": r64.tolist(),
    "arange_int_repr": repr(torch.arange(5, dtype=torch.int64)),
    "arange_range_int_repr": repr(torch.arange(2, 7, dtype=torch.int64)),
    "arange_float_repr": repr(torch.arange(0, 1, 0.25)),
    "eye_int_repr": repr(torch.eye(2, dtype=torch.int64)),
    "eye_rect_repr": repr(torch.eye(2, 3)),
    "zeros_requires_grad": torch.zeros(2, requires_grad=True).requires_grad,
    "full_like": torch.full_like(x, 3).flatten().tolist(),
    "full_like_dtype": str(torch.full_like(x, 3, dtype=torch.int64).dtype),
    "randn_like_dtype": str(torch.randn_like(n64).dtype),
    "randn_like_shape": list(torch.randn_like(n64).shape),
}))
