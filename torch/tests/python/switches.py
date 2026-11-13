"""Determinism and precision switches on RKTORCH_PARITY_DEVICE.

The twin of torch/tests/switches-parity-test.rkt (P1-P5): PyTorch's fresh
defaults, then each row's outcome under use_deterministic_algorithms(True),
strict and with warn_only=True, then each precision sequence's three
getters, and on CUDA the TF32 witnesses' distinct values. The rows mirror
determinism-rows, SEQUENCES mirrors precision-sequences, and the witnesses
mirror witness-values, all in torch/tests/private/switches.rkt.
"""
import json
import os
import re
import warnings

import torch
import torch.nn.functional as F

device = os.environ.get("RKTORCH_PARITY_DEVICE", "cpu")
KERNEL = re.compile(r"([a-z0-9_]+) does not have a deterministic implementation")
MATMUL = torch.backends.cuda.matmul
CUDNN = torch.backends.cudnn

defaults = {
    "deterministic-algorithms": torch.are_deterministic_algorithms_enabled(),
    "warn-only": torch.is_deterministic_algorithms_warn_only_enabled(),
    "cudnn-deterministic": CUDNN.deterministic,
    "cudnn-benchmark": CUDNN.benchmark,
    "float32-matmul-precision": torch.get_float32_matmul_precision(),
    "cuda-matmul-allow-tf32": MATMUL.allow_tf32,
    "cudnn-allow-tf32": CUDNN.allow_tf32,
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


SEQUENCES = {
    "high": [("precision", "high")],
    "medium": [("precision", "medium")],
    "highest": [("precision", "highest")],
    "matmul-on": [("matmul-tf32", True)],
    "matmul-on-off": [("matmul-tf32", True), ("matmul-tf32", False)],
    "high-matmul-off": [("precision", "high"), ("matmul-tf32", False)],
    "medium-matmul-on": [("precision", "medium"), ("matmul-tf32", True)],
    "medium-matmul-off": [("precision", "medium"), ("matmul-tf32", False)],
    "high-matmul-on": [("precision", "high"), ("matmul-tf32", True)],
    "matmul-on-medium": [("matmul-tf32", True), ("precision", "medium")],
    "cudnn-off": [("cudnn-tf32", False)],
    "cudnn-off-high": [("cudnn-tf32", False), ("precision", "high")],
    "high-cudnn-off": [("precision", "high"), ("cudnn-tf32", False)],
    "high-matmul-off-highest":
        [("precision", "high"), ("matmul-tf32", False), ("precision", "highest")],
}

REFUSAL = "PyTorch is checking the matmul precision"


def reset():
    torch.set_float32_matmul_precision("highest")
    MATMUL.allow_tf32 = False
    CUDNN.allow_tf32 = True


def apply_step(kind, value):
    if kind == "precision":
        torch.set_float32_matmul_precision(value)
    elif kind == "matmul-tf32":
        MATMUL.allow_tf32 = value
    else:
        CUDNN.allow_tf32 = value


def outcome(read):
    try:
        return read()
    except RuntimeError as e:
        if REFUSAL not in str(e):
            raise
        return "raise"


def sequence_outcome(steps):
    reset()
    for kind, value in steps:
        apply_step(kind, value)
    return [outcome(torch.get_float32_matmul_precision),
            outcome(lambda: MATMUL.allow_tf32),
            outcome(lambda: CUDNN.allow_tf32)]


X = 1 + 2 ** -12


def matmul_witness():
    return (torch.full((1024, 1024), X, device=device)
            @ torch.ones(1024, 1024, device=device))


def conv_witness():
    return F.conv2d(torch.full((32, 64, 56, 56), X, device=device),
                    torch.ones(128, 64, 3, 3, device=device))


MATMUL_SETTINGS = {
    "highest": lambda: torch.set_float32_matmul_precision("highest"),
    "high": lambda: torch.set_float32_matmul_precision("high"),
    "medium": lambda: torch.set_float32_matmul_precision("medium"),
    "matmul-tf32": lambda: setattr(MATMUL, "allow_tf32", True),
}

CONV_SETTINGS = {
    "cudnn-on": lambda: setattr(CUDNN, "allow_tf32", True),
    "cudnn-off": lambda: setattr(CUDNN, "allow_tf32", False),
}


def witness_under(settings, witness):
    values = {}
    for name, setting in settings.items():
        reset()
        setting()
        values[name] = torch.unique(witness()).tolist()
    return values


torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
strict_outcomes = {name: strict(run) for name, run in ROWS.items()}
torch.use_deterministic_algorithms(True, warn_only=True)
warn_outcomes = {name: warn_only(run) for name, run in ROWS.items()}
torch.use_deterministic_algorithms(False)

sequences = {name: sequence_outcome(steps) for name, steps in SEQUENCES.items()}
reset()

out = {"defaults": defaults,
       "strict": strict_outcomes,
       "warn_only": warn_outcomes,
       "sequences": sequences}
if device == "cuda":
    out["witness"] = {"matmul": witness_under(MATMUL_SETTINGS, matmul_witness),
                      "conv": witness_under(CONV_SETTINGS, conv_witness)}
    reset()

print(json.dumps(out))
