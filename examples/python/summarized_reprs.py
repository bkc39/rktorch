"""Summarized-repr parity battery (#45).

One JSON object with a list of reprs for fixed shapes covering each
summarization form: 1-d inline ellipsis (int64 width-padding), 2-d row
elision, rank-3 with only the last dim elided, rank-3 with the leading
dim elided, the just-over-threshold 1-d float case, and the bool forms
(small unsummarized + summarized) deferred from the #53 review.
"""
import json
import torch

reprs = [
    repr(torch.arange(2000)),
    repr(torch.zeros(1024, 1024)),
    repr(torch.zeros(3, 4, 500)),
    repr(torch.zeros(1024, 2, 2)),
    repr(torch.zeros(1001)),
    repr(torch.tensor([1, 2]) == 1),
    repr(torch.zeros(2000) == 1.0),
    repr(torch.arange(2000) * 1000000),
    repr(torch.arange(30) * 1000000),
    repr(torch.zeros(6, 6, 6, 5)),
    repr(torch.zeros(2, 18)),
    repr(torch.zeros(30)),
    repr(torch.full((30,), 100.0)),
    repr(torch.full((30,), float("inf"))),
    repr(torch.arange(1, 2001) * 100000000),
    repr(torch.tensor([1e10, 2.5e10, -3e-7])),
    repr(torch.tensor([1e8])),
    repr(torch.tensor([float("nan"), 5.0])),
    repr((torch.arange(2000) + 1).to(torch.float32) * 12345.6789),
    repr(torch.zeros(7, 7, 7, 7, 7)),
]
print(json.dumps({
    "reprs": reprs,
    # float64 marshal-out exactness: 2^24+1 dies in any float32 transit
    "f64_values": torch.tensor([16777217, 1],
                               dtype=torch.float64).tolist(),
}))
