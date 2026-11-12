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


def int32_scalar(v):
    return torch.tensor(v, dtype=torch.int32)


def nonfinite(ts):
    ts[0].view(-1)[0] = math.inf
    return ts


def flattened(*args):
    return [t for a in args for t in (a if isinstance(a, list) else [a])]


def as_results(r):
    return r if isinstance(r, (tuple, list)) else (r,)


HELPERS = {"torch": torch, "inf": math.inf, "int32_scalar": int32_scalar,
           "nonfinite": nonfinite, "flattened": flattened,
           "as_results": as_results}


def run(snippet):
    torch.manual_seed(0)
    scope = dict(HELPERS)
    try:
        exec(snippet, scope)
        return [flat(t) for t in scope["rs"]]
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


print(json.dumps([run(s) for s in json.load(sys.stdin)]))
