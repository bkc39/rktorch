"""F.gelu, exact (erf) form and GPT-2's tanh form.

gelu is hand-written on the C side (its kwarg-only `approximate` arg is a
`str`, outside the codegen IR/manifest), so it gets a direct check.
"""
import json
import torch
import torch.nn.functional as F

torch.manual_seed(0)
x = torch.randn(2, 3)
r = F.gelu(x)
t = F.gelu(x, approximate="tanh")
print(json.dumps({
    "shape": list(r.shape),
    "values": [float(v) for v in r.flatten().tolist()],
    "tanh": [float(v) for v in t.flatten().tolist()],
}))
