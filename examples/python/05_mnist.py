"""Parity twin of examples/racket/05-mnist.rkt: same seed, same convnet in
the same declaration order, 5 full-batch Adam steps on the committed fixture."""

import json
import os
import struct

import torch
from torch import nn

FIXTURES = os.path.join(
    os.path.dirname(__file__), "..", "..", "torch", "data", "fixtures")

DEVICE = os.environ.get("RKTORCH_PARITY_DEVICE") or "cpu"


def read_idx(path):
    with open(path, "rb") as f:
        bs = f.read()
    assert bs[0] == 0 and bs[1] == 0 and bs[2] == 8, "not a uint8 IDX buffer"
    ndim = bs[3]
    dims = [struct.unpack(">i", bs[4 + 4 * i:8 + 4 * i])[0]
            for i in range(ndim)]
    return dims, bs[4 + 4 * ndim:]


def load_fixture():
    idims, idata = read_idx(os.path.join(FIXTURES, "mnist-256-images-idx3-ubyte"))
    ldims, ldata = read_idx(os.path.join(FIXTURES, "mnist-256-labels-idx1-ubyte"))
    n, h, w = idims
    images = (torch.frombuffer(bytearray(idata), dtype=torch.uint8)
              .float().div(255.0).reshape(n, 1, h, w))
    labels = torch.frombuffer(bytearray(ldata), dtype=torch.uint8).long()
    return images, labels


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(torch.relu(self.conv(x)))


def convnet():
    return nn.Sequential(ConvBlock(1, 16), ConvBlock(16, 32), nn.Flatten(),
                         nn.Linear(800, 128), nn.ReLU(), nn.Linear(128, 10))


torch.manual_seed(0)

# construct on DEVICE: the seeded init must draw from that device's generator
with torch.device(DEVICE):
    net = convnet()
xs, ys = load_fixture()
xs, ys = xs.to(DEVICE), ys.to(DEVICE)
opt = torch.optim.Adam(net.parameters(), lr=0.001)

losses = []
for _ in range(5):
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(net(xs), ys)
    loss.backward()
    opt.step()
    losses.append(loss.item())

params = torch.cat([p.detach().flatten() for p in net.parameters()])

print(json.dumps({
    "shape": list(params.shape),
    "values": params.tolist(),
    "losses": losses,
}))
