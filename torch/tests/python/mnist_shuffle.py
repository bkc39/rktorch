"""The 05 convnet trained on shuffled minibatches of the fixture: the batch
order comes from DataLoader(generator=g), which the Racket loader replays.

Same convnet and seed as examples/python/05_mnist.py (re-declared: this
directory cannot import that one); two epochs of batch 64 with Adam.
"""
import json
import os
import struct

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "..", "data", "fixtures")


def read_idx(path):
    with open(path, "rb") as f:
        bs = f.read()
    ndim = bs[3]
    dims = [struct.unpack(">i", bs[4 + 4 * i:8 + 4 * i])[0] for i in range(ndim)]
    return dims, bs[4 + 4 * ndim:]


def load_fixture():
    idims, idata = read_idx(os.path.join(FIXTURES, "mnist-256-images-idx3-ubyte"))
    _ldims, ldata = read_idx(os.path.join(FIXTURES, "mnist-256-labels-idx1-ubyte"))
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
net = convnet()
xs, ys = load_fixture()
opt = torch.optim.Adam(net.parameters(), lr=0.001)
loader = DataLoader(TensorDataset(xs, ys), batch_size=64, shuffle=True,
                    generator=torch.Generator().manual_seed(0))

losses = []
for _ in range(2):
    for xb, yb in loader:
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(net(xb), yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())

params = torch.cat([p.detach().flatten() for p in net.parameters()])
print(json.dumps({
    "losses": losses,
    "params": params.tolist(),
}))
