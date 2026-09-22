"""Parity twin of examples/racket/12-char-rnn.rkt: same seed, the same
embedding -> LSTM -> linear model in the same declaration order, 5 full-batch
Adam steps with gradient-norm clipping on the committed fixture."""

import json
import os

import torch
from torch import nn

FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "..", "torch", "data", "fixtures",
    "heart-of-darkness-excerpt.txt")

DEVICE = os.environ.get("RKTORCH_PARITY_DEVICE") or "cpu"

BLOCK_SIZE = 16
N_EMBD = 32
HIDDEN = 64
MAX_NORM = 1.0


def load_fixture():
    with open(FIXTURE, encoding="utf-8") as f:
        text = f.read()
    vocab = sorted(set(text))
    char_to_id = {c: i for i, c in enumerate(vocab)}
    ids = torch.tensor([char_to_id[c] for c in text], dtype=torch.int64)
    b = (len(text) - 1) // BLOCK_SIZE
    xs = ids[:b * BLOCK_SIZE].reshape(b, BLOCK_SIZE)
    ys = ids[1:b * BLOCK_SIZE + 1].reshape(b, BLOCK_SIZE)
    return xs, ys, len(vocab)


class CharRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, N_EMBD)
        self.lstm = nn.LSTM(N_EMBD, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, vocab_size)

    def forward(self, idx, state=None):
        out, state = self.lstm(self.embed(idx), state)
        return self.head(out), state


xs, ys, vocab_size = load_fixture()

torch.manual_seed(0)

# construct on DEVICE: the seeded init must draw from that device's generator
with torch.device(DEVICE):
    net = CharRNN(vocab_size)
xs, ys = xs.to(DEVICE), ys.to(DEVICE)
opt = torch.optim.Adam(net.parameters(), lr=0.001)

losses = []
for _ in range(5):
    opt.zero_grad()
    logits, _ = net(xs)
    loss = nn.functional.nll_loss(
        torch.log_softmax(logits.reshape(-1, vocab_size), dim=1),
        ys.reshape(-1))
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), MAX_NORM)
    opt.step()
    losses.append(loss.item())

params = torch.cat([p.detach().flatten() for p in net.parameters()])

print(json.dumps({
    "shape": list(params.shape),
    "values": params.tolist(),
    "losses": losses,
}))
