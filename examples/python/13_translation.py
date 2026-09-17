"""Parity twin of examples/racket/13-translation.rkt: same seed, the same GRU
encoder and Bahdanau-attention GRU decoder in the same declaration order, the
same data preparation over the committed eng-fra excerpt, 5 full-batch Adam
steps with a per-batch teacher-forcing coin and gradient-norm clipping."""

import json
import os
import re
import unicodedata

import torch
from torch import nn

FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "..", "torch", "data", "fixtures",
    "eng-fra-excerpt.txt")

DEVICE = os.environ.get("RKTORCH_PARITY_DEVICE") or "cpu"

PAD, SOS, EOS = 0, 1, 2
WIDTH = 10
HIDDEN = 32
PREFIXES = ("i am ", "i m ", "he is", "he s ", "she is", "she s ",
            "you are", "you re ", "we are", "we re ", "they are", "they re ")


def normalize(s):
    s = "".join(c for c in unicodedata.normalize("NFD", s.lower().strip())
                if unicodedata.category(c) != "Mn")
    s = re.sub(r"([.!?])", r" \1", s)
    return re.sub(r"[^a-zA-Z!?]+", r" ", s).strip()


def load_pairs():
    pairs = []
    with open(FIXTURE, encoding="utf-8") as f:
        for line in f.read().strip().split("\n"):
            eng, fra = [normalize(s) for s in line.split("\t")[:2]]
            if (len(eng.split(" ")) < WIDTH and len(fra.split(" ")) < WIDTH
                    and eng.startswith(PREFIXES)):
                pairs.append((fra, eng))
    return pairs


def vocab(sentences):
    ids = {"<pad>": PAD, "<sos>": SOS, "<eos>": EOS}
    for s in sentences:
        for w in s.split(" "):
            ids.setdefault(w, len(ids))
    return ids


def encode(ids, sentences):
    rows = [[ids[w] for w in s.split(" ")] + [EOS] for s in sentences]
    return torch.tensor([r + [PAD] * (WIDTH - len(r)) for r in rows],
                        dtype=torch.int64)


class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, HIDDEN)
        self.gru = nn.GRU(HIDDEN, HIDDEN, batch_first=True)

    def forward(self, tokens):
        return self.gru(self.embed(tokens))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wa = nn.Linear(HIDDEN, HIDDEN)
        self.ua = nn.Linear(HIDDEN, HIDDEN)
        self.va = nn.Linear(HIDDEN, 1)

    def forward(self, query, keys, padding):
        scores = self.va(torch.tanh(self.wa(query) + self.ua(keys)))
        scores = scores.transpose(1, 2).masked_fill(padding, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, keys), weights


class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, HIDDEN)
        self.attend = Attention()
        self.gru = nn.GRU(2 * HIDDEN, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, vocab_size)

    def forward(self, previous, state, keys, padding):
        context, _ = self.attend(state.transpose(0, 1), keys, padding)
        output, state = self.gru(
            torch.cat([self.embed(previous), context], dim=2), state)
        return self.head(output), state


class Seq2Seq(nn.Module):
    def __init__(self, source_size, target_size):
        super().__init__()
        self.enc = Encoder(source_size)
        self.dec = Decoder(target_size)

    def forward(self, sources, targets, teacher_forcing):
        keys, state = self.enc(sources)
        padding = (sources == PAD).unsqueeze(1)
        previous = torch.full_like(sources[:, :1], SOS)
        logits = []
        for step in range(WIDTH):
            step_logits, state = self.dec(previous, state, keys, padding)
            logits.append(step_logits)
            if teacher_forcing:
                previous = targets[:, step:step + 1]
            else:
                previous = step_logits.topk(1)[1].reshape(-1, 1).detach()
        return torch.cat(logits, dim=1)


pairs = load_pairs()
source_ids = vocab([p[0] for p in pairs])
target_ids = vocab([p[1] for p in pairs])
sources = encode(source_ids, [p[0] for p in pairs])
targets = encode(target_ids, [p[1] for p in pairs])

torch.manual_seed(0)

# construct on DEVICE: the seeded init must draw from that device's generator
with torch.device(DEVICE):
    net = Seq2Seq(len(source_ids), len(target_ids))
sources, targets = sources.to(DEVICE), targets.to(DEVICE)
opt = torch.optim.Adam(net.parameters(), lr=0.001)

losses = []
for _ in range(5):
    teacher_forcing = torch.rand(1, device="cpu").item() < 0.5
    opt.zero_grad()
    logits = net(sources, targets, teacher_forcing)
    loss = nn.functional.nll_loss(
        torch.log_softmax(logits.reshape(-1, len(target_ids)), dim=1),
        targets.reshape(-1), ignore_index=PAD)
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    losses.append(loss.item())

params = torch.cat([p.detach().flatten() for p in net.parameters()])

print(json.dumps({
    "shape": list(params.shape),
    "values": params.tolist(),
    "losses": losses,
    "pairs": len(pairs),
}))
