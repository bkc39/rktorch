"""x.T reverses every axis.

T is hand-written from permute, so it gets a direct check: rank 2 through
Tensor.T itself, rank 3 through the reversed permute that defines T there
(Tensor.T on other ranks is deprecated upstream).
"""
import json
import torch

torch.manual_seed(0)
x2 = torch.randn(3, 4)
x3 = torch.randn(2, 3, 4)
r2 = x2.T
r3 = x3.permute(2, 1, 0)
print(json.dumps({
    "rank2": {"shape": list(r2.shape),
              "values": [float(v) for v in r2.flatten().tolist()]},
    "rank3": {"shape": list(r3.shape),
              "values": [float(v) for v in r3.flatten().tolist()]},
}))
