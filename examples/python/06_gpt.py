"""Parity twin of examples/racket/06-gpt.rkt: same seed, the same GPT in the
same declaration order, 5 full-batch Adam steps on the committed fixture.

The stack is the standard nn.TransformerEncoder over a pre-norm
nn.TransformerEncoderLayer with an exact-gelu MLP four times the width and
no dropout, ending in a LayerNorm, as the Racket TransformerEncoder with
#:norm #t builds it: every block a copy of the first. PyTorch's is_causal
is only a hint that the mask beside it is the causal mask, so the forward
passes both; the Racket #:causal? builds the mask itself.

The parameters are reported in the Racket model's order: each attention's
fused in_proj_weight and in_proj_bias split by rows into query, key and
value, each projection's weight followed by its bias.
"""

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
N_HEAD = 4
N_LAYER = 2


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


class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                N_EMBD, N_HEAD, dim_feedforward=4 * N_EMBD,
                activation="gelu", norm_first=True, dropout=0.0,
                batch_first=True),
            N_LAYER, norm=nn.LayerNorm(N_EMBD), enable_nested_tensor=False)
        self.head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx):
        seq_len = idx.shape[1]
        pos = torch.arange(seq_len, device=idx.device)
        causal = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=idx.device)
        h = self.tok_emb(idx) + self.pos_emb(pos)
        return self.head(self.transformer(h, mask=causal, is_causal=True))


def racket_order(net):
    for name, p in net.named_parameters():
        if name.endswith("in_proj_weight"):
            attention = net.get_submodule(name.rsplit(".", 1)[0])
            for weight, bias in zip(attention.in_proj_weight.chunk(3),
                                    attention.in_proj_bias.chunk(3)):
                yield weight
                yield bias
        elif not name.endswith("in_proj_bias"):
            yield p


xs, ys, vocab_size = load_fixture()

torch.manual_seed(0)

# construct on DEVICE: the seeded init must draw from that device's generator
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

params = torch.cat([p.detach().flatten() for p in racket_order(net)])

print(json.dumps({
    "shape": list(params.shape),
    "values": params.tolist(),
    "losses": losses,
}))
