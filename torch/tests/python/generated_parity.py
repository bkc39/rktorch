"""Expected results for torch/tests/generated-parity-test.rkt, in one process.

Reads a JSON list of snippets on stdin, one per parity case. Each runs in
a fresh namespace right after torch.manual_seed(0), as a process of its
own would, and leaves its returns in `rs`. Prints one entry per snippet:
the returns' shapes and values, or the error the snippet raised.
"""
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
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return [flat(t) for t in scope["rs"]]


print(json.dumps([run(s) for s in json.load(sys.stdin)]))
