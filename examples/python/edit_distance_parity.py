"""torchaudio.functional.edit_distance over the shared parity cases.

The Racket twin (torch/tests/audio-parity-test.rkt) runs edit-distance
on the same sequences and compares the distances.
"""
import json

import torchaudio.functional as F

CASES = [
    (list("kitten"), list("sitting")),
    (list("flaw"), list("lawn")),
    ([], []),
    ([], list("abc")),
    ("the cat sat".split(), "the bat sat on".split()),
    ("a a b a".split(), "a b a a".split()),
]

print(json.dumps({
    "distances": [F.edit_distance(a, b) for a, b in CASES],
}))
