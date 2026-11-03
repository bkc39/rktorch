"""torch.nn.init.trunc_normal_ after torch.manual_seed(3), for each case.

The twin of torch/tests/trunc-normal-test.rkt (I1), on RKTORCH_PARITY_DEVICE.
"""
import json
import os

import torch

device = os.environ.get("RKTORCH_PARITY_DEVICE", "cpu")


def drawn(dims=(64, 32), mean=0.0, std=1.0, a=-2.0, b=2.0):
    torch.manual_seed(3)
    t = torch.nn.init.trunc_normal_(
        torch.empty(*dims, device=device), mean=mean, std=std, a=a, b=b)
    return [float(v) for v in t.cpu().flatten().tolist()]


print(json.dumps({
    "std": drawn(std=0.02),
    "bounds": drawn(mean=0.01, std=0.02, a=-0.01, b=0.03),
    "tail": drawn(std=0.02, a=0.03, b=0.05),
    "edge": drawn(dims=(257,), a=0.5173725371405112, b=2.813789560067375),
}))
