"""Parity twin of examples/racket/10-dcgan.rkt: the same generator and
discriminator in the same declaration order under one seed, a reseed so
the latent draws replay on every device, and 3 full-batch steps on the
committed MNIST fixture with Adam(lr=2e-4, betas=(0.5, 0.999)); prints the
discriminator and generator losses per step interleaved and the flattened
parameters of both networks, generator first."""

import json
import os

import torch
import torch.nn.functional as F
from torch import nn

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "..", "torch",
                       "data", "fixtures", "mnist-256-images-idx3-ubyte")
DEVICE = os.environ.get("RKTORCH_PARITY_DEVICE") or "cpu"


def load_fixture():
    with open(FIXTURE, "rb") as f:
        data = f.read()
    n = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    raw = torch.frombuffer(bytearray(data[16:]), dtype=torch.uint8)
    return raw.float().div(255.0).reshape(n, 1, rows, cols)


class Generator(nn.Module):
    def __init__(self, latent=100):
        super().__init__()
        self.fc = nn.Linear(latent, 128 * 7 * 7)
        self.bn0 = nn.BatchNorm1d(128 * 7 * 7)
        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.up2 = nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1)

    def forward(self, z):
        h = F.relu(self.bn0(self.fc(z))).reshape(z.shape[0], 128, 7, 7)
        h = F.relu(self.bn1(self.up1(h)))
        return torch.tanh(self.up2(h))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.leaky_relu(self.bn(self.conv2(h)), 0.2)
        return self.fc(h.flatten(1))


torch.manual_seed(0)
xs = load_fixture().to(DEVICE)
with torch.device(DEVICE):
    gen = Generator()
    disc = Discriminator()
opt_g = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
torch.manual_seed(0)

n = xs.shape[0]
real = xs * 2.0 - 1.0
ones = torch.ones(n, 1, device=DEVICE)
zeros = torch.zeros(n, 1, device=DEVICE)
losses = []
for _ in range(3):
    z = torch.randn(n, 100, device="cpu").to(DEVICE)
    fake = gen(z)
    opt_d.zero_grad()
    d_loss = (F.binary_cross_entropy_with_logits(disc(real), ones)
              + F.binary_cross_entropy_with_logits(disc(fake.detach()), zeros))
    d_loss.backward()
    opt_d.step()
    opt_g.zero_grad()
    g_loss = F.binary_cross_entropy_with_logits(disc(fake), ones)
    g_loss.backward()
    opt_g.step()
    losses += [d_loss.item(), g_loss.item()]

params = torch.cat([p.detach().flatten()
                    for p in list(gen.parameters()) + list(disc.parameters())])
print(json.dumps({
    "shape": list(params.shape),
    "values": params.tolist(),
    "losses": losses,
}))
