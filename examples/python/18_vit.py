"""Parity twin of torch/scribblings/examples/18-vit.rkt: the same pre-norm ViT
in PyTorch's nn.TransformerEncoderLayer, started from the weights the runner
saved to RKTORCH_VIT_INIT (a safetensors file of the example's small-vit),
20 full-batch Adam steps under OneCycleLR on the committed CIFAR-10 fixture;
prints per-step losses and the parameters flattened in the order of their
sorted rktorch names."""

import json
import os
import struct

import torch
import torch.nn.functional as F
from torch import nn

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "..", "torch",
                       "vision", "fixtures", "cifar10-256.bin")
DEVICE = os.environ.get("RKTORCH_PARITY_DEVICE") or "cpu"
INIT = os.environ["RKTORCH_VIT_INIT"]
RECORD = 3073
PATCH = 4
TOKENS = 64
WIDTH = 64
DEPTH = 2
HEADS = 4
STEPS = 20


def load_fixture():
    with open(FIXTURE, "rb") as f:
        data = f.read()
    n = len(data) // RECORD
    raw = torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(n, RECORD)
    labels = raw[:, 0].long()
    images = raw[:, 1:].float().div(127.5).sub(1.0).reshape(n, 3, 32, 32)
    return images, labels


def load_safetensors(path):
    with open(path, "rb") as f:
        raw = f.read()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n])
    header.pop("__metadata__", None)
    body = raw[8 + n:]
    return {k: torch.frombuffer(bytearray(body[v["data_offsets"][0]:
                                               v["data_offsets"][1]]),
                                dtype=torch.float32).reshape(v["shape"])
            for k, v in header.items()}


class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, WIDTH, PATCH, stride=PATCH)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, WIDTH))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + TOKENS, WIDTH))
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(WIDTH, HEADS, 4 * WIDTH, dropout=0.0,
                                       activation="gelu", layer_norm_eps=1e-6,
                                       batch_first=True, norm_first=True)
            for _ in range(DEPTH))
        self.norm = nn.LayerNorm(WIDTH, eps=1e-6)
        self.head = nn.Linear(WIDTH, 10)

    def forward(self, x):
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        z = torch.cat([cls, tokens], 1) + self.pos_embed
        for block in self.blocks:
            z = block(z)
        return self.head(self.norm(z)[:, 0])


def names():
    """Each rktorch name, its twin's name, and the third of an attention's
    packed in_proj it is, or None."""
    yield "cls-token", "cls_token", None
    yield "pos-embed", "pos_embed", None
    for suffix in ("weight", "bias"):
        yield f"patch-embed.proj.{suffix}", f"patch.{suffix}", None
        yield f"norm.{suffix}", f"norm.{suffix}", None
        yield f"head.{suffix}", f"head.{suffix}", None
        for i in range(DEPTH):
            block = f"blocks.{i}."
            for ours, theirs in (("norm1", "norm1"), ("norm2", "norm2"),
                                 ("linear1", "linear1"), ("linear2", "linear2"),
                                 ("self-attn.out", "self_attn.out_proj")):
                yield f"{block}{ours}.{suffix}", f"{block}{theirs}.{suffix}", None
            for third, part in enumerate(("query", "key", "value")):
                yield (f"{block}self-attn.{part}.{suffix}",
                       f"{block}self_attn.in_proj_{suffix}", third)


def from_ours(ours):
    state = {}
    for name, theirs, third in names():
        if third is None:
            state[theirs] = ours[name]
    for name, theirs, third in names():
        if third is not None:
            state.setdefault(theirs, []).append(ours[name])
    return {k: torch.cat(v) if isinstance(v, list) else v
            for k, v in state.items()}


def flattened(net):
    theirs = dict(net.named_parameters())
    parts = {name: (theirs[twin] if third is None
                    else theirs[twin].chunk(3)[third])
             for name, twin, third in names()}
    return torch.cat([parts[name].detach().flatten() for name in sorted(parts)])


torch.manual_seed(0)

net = ViT()
net.load_state_dict(from_ours(load_safetensors(INIT)))
net.to(DEVICE)
xs, ys = load_fixture()
xs, ys = xs.to(DEVICE), ys.to(DEVICE)
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
schedule = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=1e-3, total_steps=STEPS, pct_start=0.1, cycle_momentum=False)

losses = []
for _ in range(STEPS):
    opt.zero_grad()
    loss = F.cross_entropy(net(xs), ys)
    loss.backward()
    opt.step()
    schedule.step()
    losses.append(loss.item())

params = flattened(net)

print(json.dumps({
    "shape": list(params.shape),
    "values": params.cpu().tolist(),
    "losses": losses,
}))
