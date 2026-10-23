"""A PyTorch twin that writes its meta, then fails as one out of memory does."""

import json
import sys

with open(sys.argv[1], "w") as out:
    json.dump({"meta": {"impl": "pytorch", "python": "3", "device": "cpu"}, "steps": []}, out)
sys.exit(3)
