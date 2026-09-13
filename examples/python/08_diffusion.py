"""Parity twin of examples/racket/08-diffusion.rkt: same seed, the same UNet
in the same declaration order, a linear schedule, 5 full-batch Adam steps on
the committed CIFAR-10 fixture with timesteps and noise drawn on the CPU."""

import json
import math
import os

import torch
import torch.nn.functional as F
from torch import nn

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "..", "torch",
                       "vision", "fixtures", "cifar10-256.bin")

DEVICE = os.environ.get("RKTORCH_PARITY_DEVICE") or "cpu"
RECORD = 3073


def load_fixture():
    with open(FIXTURE, "rb") as f:
        data = f.read()
    n = len(data) // RECORD
    raw = torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(n, RECORD)
    images = raw[:, 1:].float().div(127.5).sub(1.0).reshape(n, 3, 32, 32)
    return images


def linear_schedule(steps=1000, beta_start=1e-4, beta_end=0.02):
    span = max(1, steps - 1)
    betas = [beta_start + (beta_end - beta_start) * i / span for i in range(steps)]
    alphas = [1.0 - b for b in betas]
    alpha_bars = []
    acc = 1.0
    for a in alphas:
        acc *= a
        alpha_bars.append(acc)
    return torch.tensor(alpha_bars, dtype=torch.float32)


def q_sample(alpha_bars, x0, t, noise):
    a = alpha_bars[t].view(-1, 1, 1, 1)
    return a.sqrt() * x0 + (1.0 - a).sqrt() * noise


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device)
                      * (-math.log(10000.0) / half))
    angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, 4 * dim)

    def forward(self, t):
        return self.fc2(F.silu(self.fc1(sinusoidal_embedding(t, self.dim))))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(t_dim, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(temb)).view(temb.shape[0], -1, 1, 1)
        return self.conv2(F.silu(self.norm2(h))) + self.skip(x)


class UNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        t_dim = 4 * base
        self.time = TimeEmbedding(base)
        self.in_conv = nn.Conv2d(3, base, 3, padding=1)
        self.down1 = ResBlock(base, base, t_dim)
        self.pool1 = nn.Conv2d(base, base, 3, stride=2, padding=1)
        self.down2 = ResBlock(base, 2 * base, t_dim)
        self.pool2 = nn.Conv2d(2 * base, 2 * base, 3, stride=2, padding=1)
        self.mid = ResBlock(2 * base, 2 * base, t_dim)
        self.up2_conv = nn.ConvTranspose2d(2 * base, 2 * base, 4, stride=2, padding=1)
        self.up2 = ResBlock(4 * base, 2 * base, t_dim)
        self.up1_conv = nn.ConvTranspose2d(2 * base, base, 4, stride=2, padding=1)
        self.up1 = ResBlock(2 * base, base, t_dim)
        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, x, t):
        temb = self.time(t)
        h1 = self.down1(self.in_conv(x), temb)
        h2 = self.down2(self.pool1(h1), temb)
        h3 = self.mid(self.pool2(h2), temb)
        u2 = self.up2(torch.cat([self.up2_conv(h3), h2], dim=1), temb)
        u1 = self.up1(torch.cat([self.up1_conv(u2), h1], dim=1), temb)
        return self.out_conv(F.silu(self.out_norm(u1)))


torch.manual_seed(0)
xs = load_fixture().to(DEVICE)

# construct on DEVICE: the seeded init must draw from that device's generator
with torch.device(DEVICE):
    net = UNet()
    alpha_bars = linear_schedule()
opt = torch.optim.Adam(net.parameters(), lr=0.001)
steps = alpha_bars.shape[0]

losses = []
for _ in range(5):
    n = xs.shape[0]
    t = (torch.rand(n, device="cpu") * steps).long().to(DEVICE)
    noise = torch.randn(xs.shape, device="cpu").to(DEVICE)
    opt.zero_grad()
    loss = F.mse_loss(net(q_sample(alpha_bars, xs, t, noise), t), noise)
    loss.backward()
    opt.step()
    losses.append(loss.item())

params = torch.cat([p.detach().flatten() for p in net.parameters()])

print(json.dumps({
    "shape": list(params.shape),
    "values": params.tolist(),
    "losses": losses,
}))
