"""Expected results for generated-parity-test.rkt: one entry per stdin snippet."""
import json
import sys

import torch


def flat(t):
    return {"shape": list(t.shape),
            "values": [float(v) for v in t.flatten().tolist()]}


def run(snippet):
    torch.manual_seed(0)
    scope = {"torch": torch}
    try:
        exec(snippet, scope)
        return [flat(t) for t in scope["rs"]]
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


print(json.dumps([run(s) for s in json.load(sys.stdin)]))
