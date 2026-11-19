"""Numeric twin of trainer-parity-test.rkt (law L-P2 of the T0 spec, #383).

A plain PyTorch loop, no Lightning, doing what `fit` does for the test's
trainer: the convnet of examples/python/05_mnist.py on the committed MNIST
fixture in file order, batches of 40 (seven per epoch, the last of 16), SGD
with momentum, cosine annealing stepped once per epoch, gradients
accumulated over two batches, two epochs. As in Lightning, each window's
first batch zeroes the gradients after its forward, every batch backpropagates
its loss divided by the accumulation, the window's last batch (or the
epoch's) steps, and the scheduler steps after the epoch's last optimizer step.

Prints the undivided loss of every batch, and the parameters after training.

RKTORCH_TWIN_CONFIG=clip or bf16 runs trainer-clip-test.rkt's twin instead
(G3-L14 of the T2 spec, #385): a Linear(8, 4) on 32 random rows in batches of
8, SGD with momentum, two epochs, no accumulation, with clip_grad_norm_ at 1.0
between backward and step (clip), or the forward and loss under
torch.autocast("cpu", torch.bfloat16) (bf16).
"""

import json
import os
import struct

import torch
from torch import nn

FIXTURES = os.path.join(
    os.path.dirname(__file__), "..", "..", "vision", "fixtures")

DEVICE = os.environ.get("RKTORCH_PARITY_DEVICE") or "cpu"

BATCH = 40
ACCUMULATE = 2
EPOCHS = 2


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
    _, ldata = read_idx(os.path.join(FIXTURES, "mnist-256-labels-idx1-ubyte"))
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


CONFIG = os.environ.get("RKTORCH_TWIN_CONFIG")


def mnist_run():
    torch.manual_seed(0)
    with torch.device(DEVICE):
        net = convnet()
    xs, ys = load_fixture()
    xs, ys = xs.to(DEVICE), ys.to(DEVICE)
    batches = [(xs[i:i + BATCH], ys[i:i + BATCH])
               for i in range(0, len(xs), BATCH)]
    opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    losses = []
    last = len(batches) - 1
    for _ in range(EPOCHS):
        for i, (x, y) in enumerate(batches):
            loss = torch.nn.functional.cross_entropy(net(x), y)
            if i % ACCUMULATE == 0:
                opt.zero_grad(set_to_none=True)
            (loss / ACCUMULATE).backward()
            if (i + 1) % ACCUMULATE == 0 or i == last:
                opt.step()
            if i == last:
                sched.step()
            losses.append(loss.item())
    return net, losses


def config_run(config):
    torch.manual_seed(0)
    net = nn.Linear(8, 4)
    xs, ys = torch.randn(32, 8), torch.randn(32, 4)
    opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    losses = []
    for _ in range(EPOCHS):
        for i in range(0, 32, 8):
            x, y = xs[i:i + 8], ys[i:i + 8]
            with torch.autocast("cpu", dtype=torch.bfloat16,
                                enabled=config == "bf16"):
                loss = torch.nn.functional.mse_loss(net(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if config == "clip":
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
    return net, losses


net, losses = config_run(CONFIG) if CONFIG else mnist_run()
params = torch.cat([p.detach().flatten() for p in net.parameters()])

print(json.dumps({
    "shape": list(params.shape),
    "values": params.tolist(),
    "losses": losses,
}))
