"""torch.tensor's integer-dtype inference (#44).

The repr is the dtype pin: an int64 tensor prints bare integers
("tensor([[19, 22], ...])"), so a float-inferring Racket side fails the
byte-for-byte repr comparison even though the VALUES compare equal. The
matmul keeps the case honest end-to-end (integer kernels, not just
construction).
"""
import json
import torch

x = torch.tensor([[1, 2], [3, 4]])
assert x.dtype == torch.int64
r = x @ x
print(json.dumps({
    "shape": list(r.shape),
    "values": [float(v) for v in r.flatten().tolist()],
    "repr": repr(r),
}))
