"""Reference Conv-MNIST training run for examples/racket/05-mnist.rkt.

The deterministic shadow of run-example: same seed, the same LeNet-ish convnet
(layers declared in the same order so the seeded conv2d/linear inits draw
value-for-value), trained for 5 full-batch Adam steps on the committed 256-image
fixture. Full-batch (no shuffling/minibatching) keeps both languages bit-for-bit
comparable. Prints the per-step losses and the flattened post-training parameters
as {"shape": ..., "values": ..., "losses": [...]}, which torch/tests/
python-cross-test.rkt checks the Racket side against within a float tolerance.

The headline ~98% full-MNIST run lives on the Racket side (train-mnist); this twin
exists for parity, so it stays small and offline against the same fixture.
"""

import json
import os
import struct

import torch
from torch import nn

FIXTURES = os.path.join(
    os.path.dirname(__file__), "..", "..", "torch", "data", "fixtures")

# The parity twin trains on the device the cross-test pins here ("cpu" or
# "cuda"); the CUDA pass mirrors set-default-device! 'cuda on the Racket side,
# so the model is constructed on the device (seeded init uses that generator).
DEVICE = os.environ.get("RKTORCH_PARITY_DEVICE") or "cpu"


def read_idx(path):
    """Parse a uint8 IDX file into (dims, data-bytes), matching read-idx."""
    with open(path, "rb") as f:
        bs = f.read()
    assert bs[0] == 0 and bs[1] == 0 and bs[2] == 8, "not a uint8 IDX buffer"
    ndim = bs[3]
    dims = [struct.unpack(">i", bs[4 + 4 * i:8 + 4 * i])[0]
            for i in range(ndim)]
    return dims, bs[4 + 4 * ndim:]


def load_fixture():
    """(images, labels) for the committed 256-image fixture: an [N,1,H,W]
    float32 tensor scaled to [0,1] and an [N] int64 label tensor."""
    idims, idata = read_idx(os.path.join(FIXTURES, "mnist-256-images-idx3-ubyte"))
    ldims, ldata = read_idx(os.path.join(FIXTURES, "mnist-256-labels-idx1-ubyte"))
    n, h, w = idims
    images = (torch.frombuffer(bytearray(idata), dtype=torch.uint8)
              .float().div(255.0).reshape(n, 1, h, w))
    labels = torch.frombuffer(bytearray(ldata), dtype=torch.uint8).long()
    return images, labels


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3)
        self.c2 = nn.Conv2d(16, 32, 3)
        self.f1 = nn.Linear(800, 128)
        self.f2 = nn.Linear(128, 10)

    def forward(self, x):
        h = torch.max_pool2d(torch.relu(self.c1(x)), 2)
        h = torch.max_pool2d(torch.relu(self.c2(h)), 2)
        h = torch.relu(self.f1(torch.flatten(h, 1)))
        return self.f2(h)


torch.manual_seed(0)

# Construct on DEVICE so the seeded init draws from that device's generator,
# matching set-default-device! on the Racket side; data values are device-
# independent (read from the fixture) but must live where the model does.
with torch.device(DEVICE):
    net = ConvNet()
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
