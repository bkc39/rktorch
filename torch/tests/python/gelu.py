"""F.gelu, exact (erf) form.

gelu is hand-written on the C side (its kwarg-only `approximate` arg is
outside the codegen IR/manifest), so it gets a direct check.
"""
import json
import torch
import torch.nn.functional as F

torch.manual_seed(0)
x = torch.randn(2, 3)
r = F.gelu(x)
print(json.dumps({
    "shape": list(r.shape),
    "values": [float(v) for v in r.flatten().tolist()],
}))
