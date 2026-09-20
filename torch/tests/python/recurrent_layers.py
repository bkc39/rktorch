"""nn.LSTM and nn.GRU: seeded init, parameter names, forward and clipping.

The Racket LSTM and GRU layers draw their weights in nn.RNNBase's order, so
under one seed the parameters, the forward over a seeded input and the
gradients after clip_grad_norm_ must match value for value.
"""
import json
import torch
import torch.nn as nn


def flat(t):
    return [float(v) for v in t.detach().flatten().tolist()]


def case(make, with_cell):
    torch.manual_seed(0)
    m = make()
    x = torch.randn(2, 5, 3)
    out, state = m(x)
    states = list(state) if with_cell else [state]
    loss = out.sum() + sum(s.sum() for s in states)
    loss.backward()
    total = nn.utils.clip_grad_norm_(m.parameters(), 0.5)
    return {
        "names": [n for n, _ in m.named_parameters()],
        "shapes": [list(p.shape) for p in m.parameters()],
        "params": [v for p in m.parameters() for v in flat(p)],
        "out_shape": list(out.shape),
        "out": flat(out),
        "states": [flat(s) for s in states],
        "total_norm": float(total),
        "grads": [v for p in m.parameters() for v in flat(p.grad)],
    }


print(json.dumps({
    "lstm": case(lambda: nn.LSTM(3, 4, num_layers=2, bidirectional=True,
                                 batch_first=True), True),
    "gru": case(lambda: nn.GRU(3, 4, num_layers=2, bidirectional=True,
                               batch_first=True), False),
    "gru_plain": case(lambda: nn.GRU(3, 4, bias=False, batch_first=True),
                      False),
}))
