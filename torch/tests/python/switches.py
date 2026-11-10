"""Determinism switches on RKTORCH_PARITY_DEVICE.

The twin of torch/tests/switches-parity-test.rkt (P1-P3): PyTorch's fresh
defaults, then each row's outcome under use_deterministic_algorithms(True),
strict and with warn_only=True. The rows mirror determinism-rows in
torch/tests/private/switches.rkt.
"""
import json
import os
import re
import warnings

import torch
import torch.nn.functional as F

device = os.environ.get("RKTORCH_PARITY_DEVICE", "cpu")
KERNEL = re.compile(r"([a-z0-9_]+) does not have a deterministic implementation")

defaults = {
    "deterministic-algorithms": torch.are_deterministic_algorithms_enabled(),
    "warn-only": torch.is_deterministic_algorithms_warn_only_enabled(),
    "cudnn-deterministic": torch.backends.cudnn.deterministic,
    "cudnn-benchmark": torch.backends.cudnn.benchmark,
}


def bins():
    return torch.tensor([0, 3, 1, 3, 7, 2, 3, 0, 5, 3, 6, 1, 4, 3, 2, 7],
                        dtype=torch.int64, device=device)


def leaf(*dims):
    return torch.randn(*dims, device=device, requires_grad=True)


def ctc():
    log_probs = leaf(10, 2, 5).log_softmax(2)
    targets = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 1]], dtype=torch.int64,
                           device=device)
    F.ctc_loss(log_probs, targets, [10, 10], [4, 4]).backward()


def grid_sample():
    grid = torch.rand(1, 3, 3, 2, device=device) * 1.8 - 0.9
    F.grid_sample(leaf(1, 1, 4, 4), grid, align_corners=False).sum().backward()


def bicubic():
    F.interpolate(leaf(1, 1, 4, 4), size=(8, 8), mode="bicubic",
                  align_corners=False).sum().backward()


def attention():
    q, k, v = leaf(1, 2, 8, 16), leaf(1, 2, 8, 16), leaf(1, 2, 8, 16)
    F.scaled_dot_product_attention(q, k, v).sum().backward()


ROWS = {
    "adaptive-avg-pool2d-backward":
        lambda: F.adaptive_avg_pool2d(leaf(1, 2, 5, 5), (3, 3)).sum().backward(),
    "ctc-loss-backward": ctc,
    "grid-sample-backward": grid_sample,
    "upsample-bicubic2d-backward": bicubic,
    "index-add":
        lambda: torch.zeros(8, device=device).index_add_(
            0, bins(), torch.randn(16, device=device)),
    "scatter-add":
        lambda: torch.zeros(8, device=device).scatter_add_(
            0, bins(), torch.randn(16, device=device)),
    "index-select-backward":
        lambda: leaf(8, 4).index_select(0, bins()).sum().backward(),
    "embedding-backward":
        lambda: F.embedding(bins(), leaf(8, 4)).sum().backward(),
    "conv2d-backward":
        lambda: F.conv2d(torch.randn(1, 1, 8, 8, device=device),
                         leaf(2, 1, 3, 3)).sum().backward(),
    "nll-loss-forward":
        lambda: F.nll_loss(torch.randn(4, 5, device=device).log_softmax(1),
                           torch.tensor([0, 1, 2, 3], device=device)),
    "scaled-dot-product-attention-backward": attention,
}


def strict(run):
    try:
        run()
    except RuntimeError as e:
        m = KERNEL.search(str(e))
        if m is None:
            raise
        return "raise " + m.group(1)
    return "ok"


def warn_only(run):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run()
    for w in caught:
        text = str(w.message)
        if "determinis" in text:
            m = KERNEL.search(text)
            return "warn " + (m.group(1) if m else "other")
    return "ok"


torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
strict_outcomes = {name: strict(run) for name, run in ROWS.items()}
torch.use_deterministic_algorithms(True, warn_only=True)
warn_outcomes = {name: warn_only(run) for name, run in ROWS.items()}
torch.use_deterministic_algorithms(False)

print(json.dumps({"defaults": defaults,
                  "strict": strict_outcomes,
                  "warn_only": warn_outcomes}))
