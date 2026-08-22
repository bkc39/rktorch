"""Integer-dtype inference twin (#44): the bare-integer repr is the dtype
pin — a float-inferring side fails it even though the values compare equal."""
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
