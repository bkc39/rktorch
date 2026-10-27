"""The inverse-CDF truncated normal after torch.manual_seed(3), for each case.

The twin of torch/tests/trunc-normal-test.rkt (I1). This is timm's
trunc_normal_ and torch.nn.init.trunc_normal_ before PyTorch 2.14; from
2.14 PyTorch draws by rejection sampling instead, so the reference is
spelled out here rather than called.
"""
import json
import math

import torch


def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def drawn(mean=0.0, std=1.0, a=-2.0, b=2.0):
    torch.manual_seed(3)
    lo = norm_cdf((a - mean) / std)
    hi = norm_cdf((b - mean) / std)
    t = torch.empty(64, 32).uniform_(2 * lo - 1, 2 * hi - 1)
    t.erfinv_()
    t.mul_(std * math.sqrt(2.0))
    t.add_(mean)
    t.clamp_(min=a, max=b)
    return [float(v) for v in t.flatten().tolist()]


print(json.dumps({
    "std": drawn(std=0.02),
    "bounds": drawn(mean=0.01, std=0.02, a=-0.01, b=0.03),
}))
