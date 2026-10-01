"""nn.MultiheadAttention: seeded init, outputs, weights and gradients.

The twin of torch/tests/multihead-attention-parity-test.rkt. Each case
seeds, builds the layer on the CPU, so that its parameters are drawn as the
Racket layer draws its own with nothing copied across, moves it to
RKTORCH_PARITY_DEVICE, then draws its inputs, its float masks and the
loss weights in the Racket test's order.

The Racket layer keeps four Linears where PyTorch fuses the input
projection, so every parameter and gradient is reported under the Racket
names: rows [0, E), [E, 2E) and [2E, 3E) of in_proj_weight and
in_proj_bias are query, key and value (q_proj_weight, k_proj_weight and
v_proj_weight when kdim or vdim differs from E), and out_proj is out.

PyTorch's is_causal is a hint that needs attn_mask to be the causal mask;
the Racket #:causal? builds that mask, so the causal cases pass both here.
"""
import json
import os

import torch
import torch.nn as nn

device = os.environ.get("RKTORCH_PARITY_DEVICE", "cpu")


def flat(t):
    return [float(v) for v in t.detach().cpu().flatten().tolist()]


def by_name(m, of):
    e = m.embed_dim
    if m._qkv_same_embed_dim:
        packed = of(m.in_proj_weight)
        weights = [packed[i * e:(i + 1) * e] for i in range(3)]
    else:
        weights = [of(m.q_proj_weight), of(m.k_proj_weight),
                   of(m.v_proj_weight)]
    biases = ([None] * 3 if m.in_proj_bias is None
              else list(of(m.in_proj_bias).chunk(3)))
    named = {}
    for name, w, b in zip(["query", "key", "value"], weights, biases):
        named[name + ".weight"] = flat(w)
        if b is not None:
            named[name + ".bias"] = flat(b)
    named["out.weight"] = flat(of(m.out_proj.weight))
    if m.out_proj.bias is not None:
        named["out.bias"] = flat(of(m.out_proj.bias))
    return named


def leaf(*shape):
    return torch.randn(*shape).to(device).requires_grad_()


def padding(batch, length, padded):
    rows = [[False] * length for _ in range(batch)]
    for i in range(padded):
        rows[-1][length - 1 - i] = True
    return torch.tensor(rows)


def case(embed, heads, length, source, batch=2, batch_first=False,
         kdim=None, vdim=None, bias=True, dropout=0.0, cross=False,
         unbatched=False, kpm=None, mask=None, causal=False,
         need_weights=False, average=True):
    torch.manual_seed(0)
    m = nn.MultiheadAttention(embed, heads, dropout=dropout, bias=bias,
                              kdim=kdim, vdim=vdim, batch_first=batch_first)
    params = by_name(m, lambda p: p)
    m.to(device)

    def shape(n, width):
        if unbatched:
            return (n, width)
        return (batch, n, width) if batch_first else (n, batch, width)

    if cross:
        q = leaf(*shape(length, embed))
        k = leaf(*shape(source, kdim or embed))
        v = leaf(*shape(source, vdim or embed))
        leaves = [q, k, v]
    else:
        q = k = v = leaf(*shape(length, embed))
        leaves = [q]
    kpm = kpm() if kpm else None
    mask = mask() if mask else None
    if causal:
        mask = nn.Transformer.generate_square_subsequent_mask(length)
    out, weights = m(q, k, v,
                     key_padding_mask=None if kpm is None else kpm.to(device),
                     attn_mask=None if mask is None else mask.to(device),
                     is_causal=causal, need_weights=need_weights,
                     average_attn_weights=average)
    loss = (out * torch.randn(*out.shape).to(device)).sum()
    if weights is not None:
        loss = loss + (weights * torch.randn(*weights.shape).to(device)).sum()
    loss.backward()
    return {
        "params": params,
        "out_shape": list(out.shape),
        "out": flat(out),
        "weights_shape": None if weights is None else list(weights.shape),
        "weights": None if weights is None else flat(weights),
        "input_grads": [flat(t.grad) for t in leaves],
        "param_grads": by_name(m, lambda p: p.grad),
    }


print(json.dumps({
    "self_padding": case(8, 2, 5, 5, batch=3,
                         kpm=lambda: padding(3, 5, 2), need_weights=True),
    "self_padding_fused": case(8, 2, 5, 5, batch=3,
                               kpm=lambda: padding(3, 5, 2)),
    "causal": case(8, 4, 5, 5, batch_first=True, causal=True),
    "causal_weights": case(8, 4, 5, 5, batch_first=True, causal=True,
                           need_weights=True, average=False),
    "causal_padding": case(8, 4, 5, 5, batch_first=True, causal=True,
                           kpm=lambda: padding(2, 5, 1)),
    "cross": case(8, 2, 3, 4, kdim=5, vdim=6, cross=True,
                  mask=lambda: torch.randn(4, 3, 4),
                  need_weights=True, average=False),
    "cross_batch_first": case(8, 2, 3, 4, batch_first=True, kdim=5, vdim=6,
                              cross=True,
                              kpm=lambda: torch.randn(2, 4),
                              mask=lambda: torch.tril(
                                  torch.ones(3, 4, dtype=torch.bool)) == 0),
    "cross_same_width": case(8, 2, 3, 4, cross=True, need_weights=True),
    "unbatched": case(8, 2, 4, 4, unbatched=True,
                      kpm=lambda: torch.tensor([False, False, False, True]),
                      need_weights=True),
    "no_bias": case(8, 2, 3, 3, bias=False, need_weights=True),
    "dropout_fused": case(8, 2, 4, 4, dropout=0.5),
    "dropout_weights": case(8, 2, 4, 4, dropout=0.5, need_weights=True),
}))
