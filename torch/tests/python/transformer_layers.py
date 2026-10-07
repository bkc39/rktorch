"""nn.TransformerEncoderLayer, nn.TransformerDecoderLayer and their stacks.

The twin of the transformer half of torch/tests/attention-parity-test.rkt.
Each case seeds, builds the layer on the CPU, so that its parameters are
drawn as the Racket layer draws its own with nothing copied across, moves it
to RKTORCH_PARITY_DEVICE, then draws its inputs, its float masks and the
loss weights in the Racket test's order.

The Racket TransformerEncoder and TransformerDecoder built from a model
width or from a layer are nn.TransformerEncoder(layer, n, norm=...) and its
decoder twin, the layer drawn first under the same seed; built from a
procedure they are Stack, a ModuleList of layers each drawn in turn.

Every parameter and gradient is reported under the Racket names: the fused
in_proj_weight and in_proj_bias of each attention split by rows into query,
key and value, out_proj as out, and self_attn and multihead_attn spelled
self-attn and multihead-attn.

The math path is the reference: the inference fast path
(torch._transformer_encoder_layer_fwd and the nested-tensor stack) is
switched off for every case but fast_path, which runs it under no_grad in
eval mode to show the two agree.

PyTorch's is_causal is a hint that needs the mask to be the causal mask;
the Racket #:causal? builds that mask, so the causal cases pass both here.
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

device = os.environ.get("RKTORCH_PARITY_DEVICE", "cpu")
torch.backends.mha.set_fastpath_enabled(False)


def flat(t):
    return [float(v) for v in t.detach().cpu().flatten().tolist()]


def racket_name(name):
    return (name.replace("self_attn", "self-attn")
            .replace("multihead_attn", "multihead-attn")
            .replace("out_proj.", "out."))


def by_name(m, of):
    named = {}
    for name, p in m.named_parameters():
        value = of(p)
        for fused in ("in_proj_weight", "in_proj_bias"):
            if name.endswith(fused):
                prefix = racket_name(name[: -len(fused)])
                kind = "weight" if fused == "in_proj_weight" else "bias"
                for part, rows in zip(["query", "key", "value"],
                                      value.chunk(3)):
                    named[f"{prefix}{part}.{kind}"] = flat(rows)
                break
        else:
            named[racket_name(name)] = flat(value)
    return named


def gelu_tanh(x):
    return F.gelu(x, approximate="tanh")


def leaf(*shape):
    return torch.randn(*shape).to(device).requires_grad_()


def padding(batch, length, padded):
    rows = [[False] * length for _ in range(batch)]
    for i in range(padded):
        rows[-1][length - 1 - i] = True
    return torch.tensor(rows)


def later(length, source=None):
    return torch.triu(torch.ones(length, source or length, dtype=torch.bool),
                      1)


def on_device(t):
    return None if t is None else t.to(device)


def shaped(batch_first, n, length, width):
    return (n, length, width) if batch_first else (length, n, width)


def layer_options(o):
    return dict(dim_feedforward=o.get("ffn", 16),
                dropout=o.get("dropout", 0.0),
                activation=o.get("activation", F.relu),
                norm_first=o.get("norm_first", False),
                batch_first=o.get("batch_first", False),
                bias=o.get("bias", True),
                layer_norm_eps=o.get("eps", 1e-5))


class Stack(nn.Module):
    """Independently drawn layers, the twin of a stack built from a
    procedure."""

    def __init__(self, layers, norm):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm

    def forward(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x, *args, **kwargs)
        return x if self.norm is None else self.norm(x)


def final_norm(width, norm, o):
    if not norm:
        return None
    return nn.LayerNorm(width, eps=o.get("eps", 1e-5),
                        bias=o.get("bias", True))


def report(m, out, leaves):
    loss = (out * torch.randn(*out.shape).to(device)).sum()
    loss.backward()
    return {
        "out_shape": list(out.shape),
        "out": flat(out),
        "input_grads": [flat(t.grad) for t in leaves],
        "param_grads": by_name(m, lambda p: p.grad),
    }


def encoder(width=8, heads=2, length=5, batch=3, layers=None, norm=False,
            unbatched=False, kpm=None, mask=None, causal=False, train=True,
            independent=False, **o):
    torch.manual_seed(0)

    def make():
        return nn.TransformerEncoderLayer(width, heads, **layer_options(o))

    layer = make()
    if independent:
        m = Stack([layer] + [make() for _ in range(layers - 1)],
                  final_norm(width, norm, o))
    elif layers:
        m = nn.TransformerEncoder(layer, layers,
                                  norm=final_norm(width, norm, o),
                                  enable_nested_tensor=False)
    else:
        m = layer
    params = by_name(m, lambda p: p)
    m.to(device)
    m.train(train)
    batch_first = o.get("batch_first", False)
    src = leaf(*((length, width) if unbatched
                 else shaped(batch_first, batch, length, width)))
    kpm = kpm() if kpm else None
    mask = mask() if mask else None
    if causal:
        mask = nn.Transformer.generate_square_subsequent_mask(length)
    keywords = dict(src_key_padding_mask=on_device(kpm), is_causal=causal)
    if layers and not independent:
        out = m(src, mask=on_device(mask), **keywords)
    else:
        out = m(src, src_mask=on_device(mask), **keywords)
    return {"params": params, **report(m, out, [src])}


def decoder(width=8, heads=2, length=4, source=5, batch=2, layers=None,
            norm=False, tgt_kpm=None, memory_kpm=None, tgt_mask=None,
            memory_mask=None, tgt_causal=False, memory_causal=False,
            train=True, independent=False, **o):
    torch.manual_seed(0)

    def make():
        return nn.TransformerDecoderLayer(width, heads, **layer_options(o))

    layer = make()
    if independent:
        m = Stack([layer] + [make() for _ in range(layers - 1)],
                  final_norm(width, norm, o))
    elif layers:
        m = nn.TransformerDecoder(layer, layers,
                                  norm=final_norm(width, norm, o))
    else:
        m = layer
    params = by_name(m, lambda p: p)
    m.to(device)
    m.train(train)
    batch_first = o.get("batch_first", False)
    tgt = leaf(*shaped(batch_first, batch, length, width))
    memory = leaf(*shaped(batch_first, batch, source, width))
    tgt_kpm = tgt_kpm() if tgt_kpm else None
    memory_kpm = memory_kpm() if memory_kpm else None
    tgt_mask = tgt_mask() if tgt_mask else None
    memory_mask = memory_mask() if memory_mask else None
    if tgt_causal:
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(length)
    if memory_causal:
        memory_mask = later(length, source)
    out = m(tgt, memory,
            tgt_mask=on_device(tgt_mask),
            memory_mask=on_device(memory_mask),
            tgt_key_padding_mask=on_device(tgt_kpm),
            memory_key_padding_mask=on_device(memory_kpm),
            tgt_is_causal=tgt_causal,
            memory_is_causal=memory_causal)
    return {"params": params, **report(m, out, [tgt, memory])}


def fast_path():
    torch.manual_seed(0)
    m = nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.1,
                                   batch_first=True)
    params = by_name(m, lambda p: p)
    m.to(device)
    m.eval()
    src = torch.randn(3, 5, 8).to(device)
    kpm = padding(3, 5, 2).to(device)
    with torch.no_grad():
        math = m(src, src_key_padding_mask=kpm)
        torch.backends.mha.set_fastpath_enabled(True)
        try:
            fast = m(src, src_key_padding_mask=kpm)
        finally:
            torch.backends.mha.set_fastpath_enabled(False)
    return {"params": params, "out_shape": list(fast.shape), "out": flat(fast),
            "math_difference": float((fast - math).abs().max())}


print(json.dumps({
    "encoder_post_relu": encoder(kpm=lambda: padding(3, 5, 2)),
    "encoder_pre_gelu_causal": encoder(norm_first=True, activation="gelu",
                                       batch_first=True, batch=2,
                                       causal=True),
    "encoder_pre_gelu_tanh": encoder(norm_first=True, activation=gelu_tanh,
                                     batch_first=True, batch=2,
                                     mask=lambda: torch.randn(5, 5),
                                     kpm=lambda: padding(2, 5, 1)),
    "encoder_causal_padding": encoder(heads=4, causal=True,
                                      kpm=lambda: padding(3, 5, 2)),
    "encoder_eval": encoder(dropout=0.1, train=False, causal=True,
                            kpm=lambda: padding(3, 5, 1)),
    "encoder_no_bias_unbatched": encoder(bias=False, unbatched=True,
                                         eps=1e-6),
    "encoder_dropout": encoder(dropout=0.5, batch=2),
    "decoder_post_relu": decoder(tgt_causal=True,
                                 memory_kpm=lambda: padding(2, 5, 2)),
    "decoder_pre_gelu": decoder(norm_first=True, activation="gelu",
                                batch_first=True,
                                tgt_mask=lambda: torch.randn(4, 4),
                                tgt_kpm=lambda: padding(2, 4, 1),
                                memory_mask=lambda: later(4, 5)),
    "decoder_memory_causal": decoder(memory_causal=True, tgt_causal=True),
    "decoder_eval": decoder(dropout=0.1, train=False, tgt_causal=True,
                            memory_kpm=lambda: padding(2, 5, 1)),
    "decoder_dropout": decoder(dropout=0.5, norm_first=True),
    "encoder_stack": encoder(layers=3, norm=True,
                             kpm=lambda: padding(3, 5, 2)),
    "encoder_stack_pre_causal": encoder(layers=2, norm_first=True,
                                        batch_first=True, batch=2,
                                        activation=gelu_tanh, causal=True),
    "decoder_stack": decoder(layers=2, norm=True, tgt_causal=True,
                             memory_kpm=lambda: padding(2, 5, 2)),
    "decoder_stack_pre": decoder(layers=3, norm_first=True, norm=True,
                                 batch_first=True, activation="gelu",
                                 tgt_causal=True, bias=False, eps=1e-6),
    "layer_encoder": encoder(layers=3, norm=True, norm_first=True,
                             bias=False, kpm=lambda: padding(3, 5, 2)),
    "layer_decoder": decoder(layers=2, batch_first=True, tgt_causal=True,
                             memory_kpm=lambda: padding(2, 5, 2)),
    "procedure_encoder": encoder(layers=2, norm=True, independent=True,
                                 kpm=lambda: padding(3, 5, 2)),
    "procedure_decoder": decoder(layers=2, independent=True,
                                 norm_first=True, activation=gelu_tanh,
                                 batch_first=True, tgt_causal=True,
                                 memory_kpm=lambda: padding(2, 5, 1)),
    "fast_path": fast_path(),
}))
