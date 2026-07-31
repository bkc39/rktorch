"""F.max_pool2d / F.avg_pool2d with stride defaulted to kernel size.

The promoted Racket wrappers default #:stride to kernel-size (PyTorch's
stride=None); the generated battery hits the raw bindings, so this
checks the facade default directly.
"""
import json
import torch
import torch.nn.functional as F

torch.manual_seed(0)
x = torch.randn(1, 1, 4, 4)
mp = F.max_pool2d(x, 2)
ap = F.avg_pool2d(x, 2)
print(json.dumps({
    "mp": [float(v) for v in mp.flatten().tolist()],
    "ap": [float(v) for v in ap.flatten().tolist()],
}))
