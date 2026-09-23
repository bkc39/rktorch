"""torch.autocast on the CPU in bfloat16: a matmul's dtype and values, a
Linear's forward under the cast with float32 gradients outside it, and
the state before and after the extent.
"""
import json
import torch
import torch.nn as nn

torch.manual_seed(0)
a = torch.randn(3, 4)
b = torch.randn(4, 2)
before = torch.is_autocast_enabled("cpu")
with torch.autocast("cpu", dtype=torch.bfloat16):
    inside = torch.is_autocast_enabled("cpu")
    prod = a @ b
after = torch.is_autocast_enabled("cpu")

torch.manual_seed(1)
lin = nn.Linear(4, 2)
x = torch.randn(8, 4)
with torch.autocast("cpu", dtype=torch.bfloat16):
    y = lin(x)
    loss = (y * y).mean()
loss.backward()
print(json.dumps({
    "before": before,
    "inside": inside,
    "after": after,
    "prod_dtype": str(prod.dtype),
    "prod_values": prod.to(torch.float32).flatten().tolist(),
    "y_dtype": str(y.dtype),
    "loss": float(loss),
    "grad_dtype": str(lin.weight.grad.dtype),
    "grad_values": lin.weight.grad.flatten().tolist(),
}))
