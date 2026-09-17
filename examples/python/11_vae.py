"""Parity twin of examples/racket/11-vae.rkt: the same linear VAE in the
same declaration order under one seed, a reseed so the noise draws replay
on every device, 5 full-batch Adam steps on the committed MNIST fixture
with the reference loss divided by the batch size; prints per-step losses
and the flattened parameters."""

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


class VAE(nn.Module):
    def __init__(self, latent=20, hidden=400):
        super().__init__()
        self.enc = nn.Linear(784, hidden)
        self.mu_head = nn.Linear(hidden, latent)
        self.logvar_head = nn.Linear(hidden, latent)
        self.dec1 = nn.Linear(latent, hidden)
        self.dec2 = nn.Linear(hidden, 784)

    def forward(self, x, eps):
        h = F.relu(self.enc(x.flatten(1)))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = mu + eps * torch.exp(0.5 * logvar)
        return self.dec2(F.relu(self.dec1(z))), mu, logvar


def vae_loss(logits, x, mu, logvar):
    n = x.shape[0]
    recon = F.binary_cross_entropy_with_logits(logits, x.flatten(1)) * 784.0
    kl = torch.sum(1.0 + logvar - mu * mu - logvar.exp()) * (-0.5 / n)
    return recon + kl


torch.manual_seed(0)
xs = load_fixture().to(DEVICE)
with torch.device(DEVICE):
    net = VAE()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
torch.manual_seed(0)

losses = []
for _ in range(5):
    eps = torch.randn(xs.shape[0], 20, device="cpu").to(DEVICE)
    opt.zero_grad()
    logits, mu, logvar = net(xs, eps)
    loss = vae_loss(logits, xs, mu, logvar)
    loss.backward()
    opt.step()
    losses.append(loss.item())

params = torch.cat([p.detach().flatten() for p in net.parameters()])
print(json.dumps({
    "shape": list(params.shape),
    "values": params.tolist(),
    "losses": losses,
}))
