"""Parity twin of torch/scribblings/examples/17-workers.rkt: the plain-PGM
encoding of the committed fixture, byte for byte; the jitter at fixed angles
and shifts, as torch's affine_grid and grid_sample compute it (bilinear, zero
outside the image, align_corners=True so the centre is pixel 13.5); then the
convnet of 05_mnist.py, same seed, 5 full-batch Adam steps on the fixture
jittered image by image.

The angles and shifts are fixed here, not drawn: the loader's batch order and
its per-worker generators are dataloader_twin.py's subject, and the Racket
PRNG behind jitter-draws has no Python counterpart.
"""

import json
import math
import os
import struct

import torch
import torch.nn.functional as F
from torch import nn

FIXTURES = os.path.join(
    os.path.dirname(__file__), "..", "..", "torch", "vision", "fixtures")

DEVICE = os.environ.get("RKTORCH_PARITY_DEVICE") or "cpu"

SIDE = 28

# (angle in radians, dx, dy): the identity, half-pixel and fractional shifts,
# a rotation each way, both together, and a half turn
CASES = [
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 1.25, -0.75),
    (0.2, 0.0, 0.0),
    (-0.26, 1.7, -1.3),
    (0.1, -0.4, 0.9),
    (math.pi, 0.0, 0.0),
]

# image i of the training batch is jittered by CASES[i % len(CASES)]; the
# first JITTERED images travel back for the pixel-level comparison
JITTERED = 2 * len(CASES)


def read_idx(path):
    with open(path, "rb") as f:
        bs = f.read()
    assert bs[0] == 0 and bs[1] == 0 and bs[2] == 8, "not a uint8 IDX buffer"
    ndim = bs[3]
    dims = [struct.unpack(">i", bs[4 + 4 * i:8 + 4 * i])[0]
            for i in range(ndim)]
    return dims, bs[4 + 4 * ndim:]


def load_fixture():
    """The fixture as the Racket side encodes it: scaled to [0, 1] by
    load-mnist-fixture, then back to bytes by images->pgms."""
    idims, idata = read_idx(os.path.join(FIXTURES, "mnist-256-images-idx3-ubyte"))
    _, ldata = read_idx(os.path.join(FIXTURES, "mnist-256-labels-idx1-ubyte"))
    n = idims[0]
    scaled = torch.frombuffer(bytearray(idata), dtype=torch.uint8).float() / 255.0
    pixels = (scaled * 255 + 0.5).to(torch.uint8).reshape(n, SIDE * SIDE)
    labels = torch.frombuffer(bytearray(ldata), dtype=torch.uint8).long()
    return [bytes(row.tolist()) for row in pixels], labels


def every_value():
    """A pattern with no zero border, so sampling outside the image shows."""
    return bytes((7 * k) % 256 for k in range(SIDE * SIDE))


def pgm_encode(pixels, sep=" ", eol="\n", header="P2\n28 28\n255\n"):
    rows = (sep.join(str(v) for v in pixels[r * SIDE:(r + 1) * SIDE])
            for r in range(SIDE))
    return (header + "".join(row + eol for row in rows)).encode("ascii")


def pgm_decode(bs):
    fields = bs.split()
    assert fields[:4] == [b"P2", b"28", b"28", b"255"], fields[:4]
    return bytes(int(v) for v in fields[4:])


def jitter(pixels, angle, dx, dy):
    """Output pixel p samples the source at R(-angle)(p - c - (dx, dy)) + c."""
    c, s = math.cos(angle), math.sin(angle)
    half = (SIDE - 1) / 2
    theta = torch.tensor([[[c, s, -(c * dx + s * dy) / half],
                           [-s, c, (s * dx - c * dy) / half]]],
                         dtype=torch.float64)
    grid = F.affine_grid(theta, [1, 1, SIDE, SIDE], align_corners=True)
    image = torch.tensor(list(pixels), dtype=torch.float64).reshape(
        1, 1, SIDE, SIDE)
    out = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros",
                        align_corners=True)
    return (out / 255.0).float().reshape(SIDE * SIDE)


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


images, ys = load_fixture()
pattern = every_value()

pgms = [pgm_encode(p) for p in [pattern] + images]
assert all(pgm_decode(bs) == p for bs, p in zip(pgms, [pattern] + images))
loose = [pgm_encode(p, sep=" \t ", eol="\r\n", header="P2\t28\n 28  255\r\n")
         for p in [pattern] + images[:2]]

pattern_jitters = [jitter(pattern, *case) for case in CASES]
image_jitters = [jitter(p, *CASES[i % len(CASES)])
                 for i, p in enumerate(images)]

torch.manual_seed(0)

# construct on DEVICE: the seeded init must draw from that device's generator
with torch.device(DEVICE):
    net = convnet()
xs = torch.stack(image_jitters).reshape(len(images), 1, SIDE, SIDE).to(DEVICE)
ys = ys.to(DEVICE)
opt = torch.optim.Adam(net.parameters(), lr=0.001)

losses = []
for _ in range(5):
    opt.zero_grad()
    loss = F.cross_entropy(net(xs), ys)
    loss.backward()
    opt.step()
    losses.append(loss.item())

params = torch.cat([p.detach().flatten() for p in net.parameters()])

print(json.dumps({
    "cases": [list(case) for case in CASES],
    "pgm": b"".join(pgms).hex(),
    "loose_pgms": [bs.hex() for bs in loose],
    "pattern_jitters": [t.tolist() for t in pattern_jitters],
    "image_jitters": [t.tolist() for t in image_jitters[:JITTERED]],
    "shape": list(params.shape),
    "values": params.tolist(),
    "losses": losses,
}))
