"""Indexing twin (#46): each entry mirrors one ref spec form in
torch/tests/python-cross-test.rkt, in order — [shape, flat values] per
tensor result, a bare number for scalar results."""
import json
import torch


def form(x):
    return [list(x.shape), [float(v) for v in x.flatten().tolist()]]


t = torch.tensor([[1, 2, 3], [4, 5, 6]])
c = torch.arange(6, dtype=torch.float32)
cube = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

forms = [
    form(t[0]),
    form(t[1:3]),
    form(c[1:5:2]),
    form(c[::2]),
    form(t[:, 1:]),
    form(t[..., 0]),
    form(t[:, None]),
    form(cube[1, ..., 0]),
    form(t[t > 4]),
    form(t[torch.tensor([False, True])]),
    form(t[[-1, 0]]),
    form(cube[torch.tensor([[True, False], [False, True]])]),
    form(t[[0, 0, 1]]),
    form(t[[1, 0], 0]),
    form(c[torch.tensor([4, 0])]),
    float(t[1, 2]),
    form(torch.where(t > 4)[0]),
    form(torch.where(t > 4)[1]),
    form(torch.gather(t, 1, torch.tensor([[0, 2], [1, 0]]))),
    form(torch.take(t, torch.tensor([0, 5, 3]))),
    form(torch.take_along_dim(t, torch.tensor([5, 0, 2]))),
    form(torch.take(t, torch.tensor([-1, 0, -6]))),
    form(torch.where(torch.tensor(True))[0]),
    form(torch.where(torch.tensor(False))[0]),
]
print(json.dumps({"forms": forms}))
