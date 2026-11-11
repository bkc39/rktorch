"""Expected results for generated-parity-test.rkt: one entry per stdin snippet.
A non-finite value travels as a Racket-readable string: JSON has none."""
import json
import math
import sys

import torch


def rkt(v):
    v = float(v)
    if math.isnan(v):
        return "+nan.0"
    if math.isinf(v):
        return "+inf.0" if v > 0 else "-inf.0"
    return v


def flat(t):
    return {"shape": list(t.shape),
            "values": [rkt(v) for v in t.flatten().tolist()]}


def run(snippet):
    torch.manual_seed(0)
    scope = {"torch": torch}
    try:
        exec(snippet, scope)
        return [flat(t) for t in scope["rs"]]
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


print(json.dumps([run(s) for s in json.load(sys.stdin)]))
