"""Loss trajectories of the base-16 ResNet on the fixture for a few recipes,
to tell chaos from an update bug when the Racket side diverges."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "python"))
os.environ.setdefault("RKTORCH_PARITY_DEVICE", "cpu")

import torch
import torch.nn.functional as F

import importlib.util
spec = importlib.util.spec_from_file_location(
    "resnet_twin",
    os.path.join(os.path.dirname(__file__), "..", "examples", "python", "09_resnet.py"))
# the twin runs its training at import; we only want its classes and loader
src = open(spec.origin).read().split("torch.manual_seed(0)\nxs, ys = load_fixture()")[0]
ns = {"__file__": spec.origin}
exec(compile(src, spec.origin, "exec"), ns)

recipes = {
    "full": dict(lr=0.05, momentum=0.9, weight_decay=5e-4),
    "plain": dict(lr=0.05, momentum=0.0, weight_decay=0.0),
    "gentle": dict(lr=0.005, momentum=0.9, weight_decay=5e-4),
    "momentum_only": dict(lr=0.05, momentum=0.9, weight_decay=0.0),
}
out = {}
for name, kw in recipes.items():
    torch.manual_seed(0)
    xs, ys = ns["load_fixture"]()
    net = ns["ResNet"](base=16)
    opt = torch.optim.SGD(net.parameters(), **kw)
    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = F.cross_entropy(net(xs), ys)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    out[name] = losses
print(json.dumps(out))
