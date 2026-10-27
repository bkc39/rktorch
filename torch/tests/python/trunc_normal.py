"""torch.nn.init.trunc_normal_ after torch.manual_seed(3), for each case.

The twin of torch/tests/trunc-normal-test.rkt (I1).
"""
import json

import torch


def drawn(mean=0.0, std=1.0, a=-2.0, b=2.0):
    torch.manual_seed(3)
    t = torch.nn.init.trunc_normal_(
        torch.empty(64, 32), mean=mean, std=std, a=a, b=b)
    return [float(v) for v in t.flatten().tolist()]


print(json.dumps({
    "std": drawn(std=0.02),
    "bounds": drawn(mean=0.01, std=0.02, a=-0.01, b=0.03),
    "tail": drawn(std=0.02, a=0.03, b=0.05),
}))
