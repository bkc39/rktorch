"""Three Adam steps on two parameters where the second step's loss leaves b
out: zero_grad() drops b's gradient, so that step skips b, while
zero_grad(set_to_none=False) leaves a zero one that momentum still moves."""
import json
import torch


def trained(set_to_none):
    a = torch.tensor([1.0, -2.0], requires_grad=True)
    b = torch.tensor([0.5, 3.0], requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=0.1)
    for uses_b in (True, False, True):
        opt.zero_grad(set_to_none=set_to_none)
        loss = (a * a).sum()
        if uses_b:
            loss = loss + (b * b).sum()
        loss.backward()
        opt.step()
    return torch.cat([a.detach(), b.detach()]).tolist()


print(json.dumps({"none": trained(True), "zero": trained(False)}))
