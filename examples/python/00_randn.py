"""Reference draw for the Racket parity cross-test.

Seeds the global RNG and samples a 2x2 standard-normal tensor, then prints it as
{"shape": [...], "values": [...]} (row-major). torchrkt's `examples/racket/
00-randn.rkt` performs the identical computation; `torchrkt/tests/
python-cross-test.rkt` compares the two within a float tolerance.
"""

import json

import torch

torch.manual_seed(0)
t = torch.randn(2, 2)

print(json.dumps({
    "shape": list(t.shape),
    "values": t.flatten().tolist(),
    "repr": repr(t),
}))
