"""nn.BatchNorm2d: a training forward, the running statistics it leaves
behind, and the eval forward that uses them.

BatchNorm2d's init is deterministic (ones/zeros), so the forward and the
buffers after one step are the meaningful parity checks for the Racket
BatchNorm2d layer.
"""
import json
import torch
import torch.nn as nn

torch.manual_seed(0)
m = nn.BatchNorm2d(3)
x = torch.randn(2, 3, 4, 4)
y = m(x)
m.eval()
z = m(x)
print(json.dumps({
    "shape": list(y.shape),
    "values": [float(v) for v in y.detach().flatten().tolist()],
    "running_mean": [float(v) for v in m.running_mean.tolist()],
    "running_var": [float(v) for v in m.running_var.tolist()],
    "num_batches_tracked": int(m.num_batches_tracked),
    "eval_values": [float(v) for v in z.detach().flatten().tolist()],
}))
