"""Reference char-GPT training run for examples/racket/06-gpt.rkt.

The deterministic shadow of run-example: same seed, the same pre-norm
decoder-only transformer (modules declared in the same order so the seeded
embedding/linear inits draw value-for-value), trained for 5 full-batch Adam
steps on the committed 841-char Heart of Darkness fixture. Full-batch, no
shuffling, greedy everything: both languages stay bit-comparable. Prints the
per-step losses and the flattened post-training parameters as
{"shape": ..., "values": ..., "losses": [...]}, which
torch/tests/python-cross-test.rkt checks the Racket side against within a
float tolerance.

The headline full-novella run lives on the Racket side (train-novel); this
twin exists for parity, so it stays small and offline against the fixture.
"""

import json
import math
import os

import torch
from torch import nn

FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "..", "torch", "data", "fixtures",
    "heart-of-darkness-excerpt.txt")

# The parity twin trains on the device the cross-test pins here ("cpu" or
# "cuda"); the CUDA pass mirrors with-default-device 'cuda on the Racket
# side, so the model is constructed on the device (seeded init uses that
# device's generator).
DEVICE = os.environ.get("RKTORCH_PARITY_DEVICE") or "cpu"

BLOCK_SIZE = 16   # fixture-block-size
N_EMBD = 32       # the gpt module's fixture-scale defaults
N_HEAD = 4
N_LAYER = 2


def load_fixture():
    """(xs, ys, vocab_size): the fixture text char-encoded against its own
    sorted-unique vocab and split into contiguous next-char blocks, exactly
    like text->vocab + encode + contiguous-blocks."""
    with open(FIXTURE, encoding="utf-8") as f:
        text = f.read()
    vocab = sorted(set(text))
    char_to_id = {c: i for i, c in enumerate(vocab)}
    ids = torch.tensor([char_to_id[c] for c in text], dtype=torch.int64)
    b = (len(text) - 1) // BLOCK_SIZE
    xs = ids[:b * BLOCK_SIZE].reshape(b, BLOCK_SIZE)
    ys = ids[1:b * BLOCK_SIZE + 1].reshape(b, BLOCK_SIZE)
    return xs, ys, len(vocab)


class Block(nn.Module):
    """Pre-norm transformer block; submodule declaration order matches
    gpt-block so the seeded inits draw in the same sequence."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.wq = nn.Linear(n_embd, n_embd)
        self.wk = nn.Linear(n_embd, n_embd)
        self.wv = nn.Linear(n_embd, n_embd)
        self.wo = nn.Linear(n_embd, n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        batch, seq_len, n_embd = x.shape
        head_dim = n_embd // self.n_head

        def split_heads(m):
            return m.reshape(batch, seq_len, self.n_head,
                             head_dim).transpose(1, 2)

        xn = self.ln1(x)
        q, k, v = split_heads(self.wq(xn)), split_heads(self.wk(xn)), \
            split_heads(self.wv(xn))
        scores = q @ k.transpose(2, 3) / math.sqrt(head_dim)
        # build the mask on the input's device (the Racket side inherits it
        # from with-default-device; forward-time torch.ones does not).
        causal = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device)) == 0
        att = torch.softmax(scores.masked_fill(causal, float("-inf")), -1)
        ctx = (att @ v).transpose(1, 2).reshape(batch, seq_len, n_embd)
        x = x + self.wo(ctx)
        # F.gelu's default is the exact erf form, matching Racket's gelu.
        return x + self.fc2(nn.functional.gelu(self.fc1(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(
            *[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx):
        seq_len = idx.shape[1]
        pos = torch.arange(seq_len, device=idx.device)
        h = self.tok_emb(idx) + self.pos_emb(pos)
        return self.head(self.ln_f(self.blocks(h)))


xs, ys, vocab_size = load_fixture()

torch.manual_seed(0)

# Construct on DEVICE so the seeded init draws from that device's generator,
# matching with-default-device on the Racket side; the data is device-
# independent but must live where the model does.
with torch.device(DEVICE):
    net = GPT(vocab_size)
xs, ys = xs.to(DEVICE), ys.to(DEVICE)
opt = torch.optim.Adam(net.parameters(), lr=0.001)

losses = []
for _ in range(5):
    opt.zero_grad()
    logits = net(xs)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size), ys.reshape(-1))
    loss.backward()
    opt.step()
    losses.append(loss.item())

params = torch.cat([p.detach().flatten() for p in net.parameters()])

print(json.dumps({
    "shape": list(params.shape),
    "values": params.tolist(),
    "losses": losses,
}))
