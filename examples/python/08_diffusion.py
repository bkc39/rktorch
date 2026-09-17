"""Parity twin of examples/racket/08-diffusion.rkt: same seed, the same UNet
in the same declaration order (a small configuration of the DDPM CIFAR-10
network: base 64, two levels, attention at 16x16, no dropout), a linear
schedule, 5 full-batch Adam steps on the committed CIFAR-10 fixture with
timesteps and noise drawn on the CPU."""

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
GROUPS = 32


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
    def __init__(self, cin, cout, t_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(GROUPS, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(t_dim, cout)
        self.norm2 = nn.GroupNorm(GROUPS, cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(temb)).view(temb.shape[0], -1, 1, 1)
        return self.conv2(self.drop(F.silu(self.norm2(h)))) + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(GROUPS, channels)
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        n, c, h, w = x.shape
        tokens = self.norm(x).reshape(n, c, h * w).transpose(1, 2)
        scores = torch.bmm(self.q(tokens), self.k(tokens).transpose(1, 2)) * (c ** -0.5)
        mixed = self.proj(torch.bmm(torch.softmax(scores, dim=-1), self.v(tokens)))
        return x + mixed.transpose(1, 2).reshape(n, c, h, w)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x, temb):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class Stage(nn.Module):
    def __init__(self, res, attn):
        super().__init__()
        self.res = res
        self.attn = attn

    def forward(self, x, temb):
        h = self.res(x, temb)
        return self.attn(h) if self.attn is not None else h


class UNet(nn.Module):
    def __init__(self, base=128, mults=(1, 2, 2, 2), blocks=2, attention=(16,),
                 dropout=0.1, classes=None):
        super().__init__()
        t_dim = 4 * base
        levels = len(mults)

        def stage(cin, cout, res):
            # the ResBlock draws before the attention block, as the Racket
            # constructor does: RNG order is the parity contract
            res_block = ResBlock(cin, cout, t_dim, dropout)
            attn = AttentionBlock(cout) if res in attention else None
            return Stage(res_block, attn)

        self.time = TimeEmbedding(base)
        self.classes = nn.Embedding(classes + 1, t_dim) if classes else None
        self.in_conv = nn.Conv2d(3, base, 3, padding=1)
        downs = []
        skips = [base]
        cin = base
        for i in range(levels):
            cout = base * mults[i]
            res = 32 // (2 ** i)
            for _ in range(blocks):
                downs.append(stage(cin, cout, res))
                skips.append(cout)
                cin = cout
            if i != levels - 1:
                downs.append(Downsample(cout))
                skips.append(cout)
        self.downs = nn.ModuleList(downs)
        self.mid1 = ResBlock(cin, cin, t_dim, dropout)
        self.mid_attn = AttentionBlock(cin)
        self.mid2 = ResBlock(cin, cin, t_dim, dropout)
        ups = []
        for i in reversed(range(levels)):
            cout = base * mults[i]
            res = 32 // (2 ** i)
            for _ in range(blocks + 1):
                ups.append(stage(cin + skips.pop(), cout, res))
                cin = cout
            if i != 0:
                ups.append(Upsample(cout))
        self.ups = nn.ModuleList(ups)
        self.out_norm = nn.GroupNorm(GROUPS, base)
        self.out_conv = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, x, t, y=None):
        temb = self.time(t)
        if self.classes is not None:
            temb = temb + self.classes(y)
        h = self.in_conv(x)
        skips = [h]
        for stage in self.downs:
            h = stage(h, temb)
            skips.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
        for stage in self.ups:
            if isinstance(stage, Upsample):
                h = stage(h)
            else:
                h = stage(torch.cat([h, skips.pop()], dim=1), temb)
        return self.out_conv(F.silu(self.out_norm(h)))


torch.manual_seed(0)
xs = load_fixture().to(DEVICE)

# construct on DEVICE: the seeded init must draw from that device's generator
with torch.device(DEVICE):
    net = UNet(base=64, mults=(1, 2), blocks=1, attention=(16,), dropout=0.0)
    alpha_bars = linear_schedule()
opt = torch.optim.Adam(net.parameters(), lr=0.001)
steps = alpha_bars.shape[0]

# building the net consumed the CPU stream only on a CPU run; reseed so
# the timestep and noise draws below replay on every device
torch.manual_seed(0)

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
