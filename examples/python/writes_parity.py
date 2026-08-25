"""Write twin (#73 Family 3): each entry mirrors one ref! / write-op
form in torch/tests/python-cross-test.rkt, in order — [shape, flat
values] of the mutated target."""
import json
import torch


def form(x):
    return [list(x.shape), [float(v) for v in x.flatten().tolist()]]


def fresh():
    return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


forms = []

t = fresh(); t[0] = 9; forms.append(form(t))
t = fresh(); t[1, 2] = 0; forms.append(form(t))
t = fresh(); t[-1, -1] = 0; forms.append(form(t))
t = fresh(); t[1:3] = 7; forms.append(form(t))
t = fresh(); t[:, ::2] = 0; forms.append(form(t))
t = fresh(); t[..., 0] = 5; forms.append(form(t))
t = fresh(); t[:, 1] = torch.tensor([8.0, 9.0]); forms.append(form(t))
t = fresh(); t[0] = torch.tensor([7.0, 8.0, 9.0]); forms.append(form(t))
z = torch.tensor(5.0); z[...] = 7; forms.append(form(z.reshape(1)))
t = fresh(); t[t > 4] = -1; forms.append(form(t))
t = fresh(); t[t > 4] = torch.tensor(0.0); forms.append(form(t))
t = fresh(); t[t > 3] = torch.tensor([7.0, 8.0, 9.0]); forms.append(form(t))
t = fresh(); t[torch.tensor([True, False])] = 0; forms.append(form(t))
t = fresh()
t[torch.tensor([False, True])] = torch.tensor([7.0, 8.0, 9.0])
forms.append(form(t))
t = fresh(); t[torch.tensor([True, False])] = torch.tensor([7.0])
forms.append(form(t))
t = fresh(); t[torch.tensor([True, True])] = torch.tensor([[7.0], [8.0]])
forms.append(form(t))
t = fresh(); t.index_fill_(0, torch.tensor([1]), 3.5); forms.append(form(t))
t = fresh()
t.index_copy_(0, torch.tensor([0]), torch.tensor([[9.0, 9.0, 9.0]]))
forms.append(form(t))
t = fresh()
t.index_add_(0, torch.tensor([0]), torch.tensor([[1.0, 1.0, 1.0]]), alpha=2)
forms.append(form(t))
t = fresh(); t.scatter_(1, torch.tensor([[0], [2]]), 5.0); forms.append(form(t))
t = fresh()
t.scatter_(1, torch.tensor([[0], [2]]), torch.tensor([[70.0], [80.0]]))
forms.append(form(t))
t = fresh()
t.scatter_add_(1, torch.tensor([[0], [0]]), torch.tensor([[1.0], [1.0]]))
forms.append(form(t))
t = fresh(); t.masked_fill_(t > 5, 0); forms.append(form(t))
t = fresh(); t.masked_scatter_(t > 4, torch.tensor([9.0, 10.0]))
forms.append(form(t))
print(json.dumps({"forms": forms}))
