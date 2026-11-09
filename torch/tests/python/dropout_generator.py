"""Dropout masks for torch/tests/dropout-generator-test.rkt.

"global" is F.dropout after torch.manual_seed(7), the default generator.
"generator" is the mask rktorch draws from a generator seeded 7, written
in PyTorch: dropout's own arithmetic with torch.rand in bernoulli_'s place,
since F.dropout takes no generator.
"""
import json

import torch
import torch.nn.functional as F

p = 0.3
keep = 1 - p
x = torch.arange(1, 257, dtype=torch.float32).reshape(4, 64)


def flat(t):
    return [float(v) for v in t.flatten().tolist()]


torch.manual_seed(7)
global_stream = F.dropout(x, p)

g = torch.Generator().manual_seed(7)
noise = (torch.rand(x.shape, dtype=x.dtype, generator=g) < keep).to(x.dtype)
generator = x * noise.div_(keep)

print(json.dumps({"global": flat(global_stream), "generator": flat(generator)}))
