"""Lightning 2.6.5 twin of torch/scribblings/examples/20-fashion-mnist.rkt.

The chapter's convnet, AdamW over two parameter groups, a linear warmup into a
cosine schedule stepped per optimizer step, two-batch accumulation, norm
clipping at 0.5, and the precision RKTORCH_PARITY_PRECISION names, on
RKTORCH_PARITY_DEVICE. RKTORCH_TWIN_MODE=fixture trains on the committed
fixture and prints a step-by-step trace; full trains on the .gz archives in
RKTORCH_FASHION_MNIST_DIR and adds one row per epoch. EPOCHS sets the epochs.
A non-finite float is printed as the string "inf", "-inf" or "nan".

RKTORCH_TWIN_MODE=golden traces the fixture fits in 32-true and bf16-mixed
on one CPU thread, the count the test cells run with, and writes them, each
tensor of the initial and final states summarized, to
examples/test/20-fashion-mnist-lightning.rktd, or the path given as the first
argument; examples/test/20-fashion-mnist.rkt compares Racket's fits with it,
and regenerates it where Lightning is installed.
"""

import gzip
import json
import math
import os
import platform
import struct
import sys
import tempfile
import time
import warnings

import lightning
import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers.logger import Logger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "..", "torch", "vision",
                        "fixtures", "fashion-mnist")

DEVICE = os.environ.get("RKTORCH_PARITY_DEVICE") or "cpu"
PRECISION = os.environ.get("RKTORCH_PARITY_PRECISION") or "32-true"
MODE = os.environ.get("RKTORCH_TWIN_MODE") or "fixture"
EPOCHS = int(os.environ.get("EPOCHS") or "2")

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test",
                      "20-fashion-mnist-lightning.rktd")
GOLDEN_PRECISIONS = ("32-true", "bf16-mixed")

if DEVICE not in ("cpu", "cuda"):
    raise SystemExit(f"RKTORCH_PARITY_DEVICE={DEVICE}: the twin runs on cpu or cuda")

SEED = 0
LR = 2e-3
WEIGHT_DECAY = 0.05
CLIP = 0.5
ACCUMULATE = 2
BATCH = {"fixture": 32, "golden": 32, "full": 128}[MODE]
VAL_BATCH = {"fixture": 64, "golden": 64, "full": 1000}[MODE]


def read_idx(bs):
    assert bs[0] == 0 and bs[1] == 0 and bs[2] == 8, "not a uint8 IDX buffer"
    ndim = bs[3]
    dims = [struct.unpack(">I", bs[4 + 4 * i:8 + 4 * i])[0] for i in range(ndim)]
    return dims, bs[4 + 4 * ndim:]


def tensors(images, labels):
    dims, data = read_idx(images)
    _, ldata = read_idx(labels)
    xs = (torch.frombuffer(bytearray(data), dtype=torch.uint8)
          .to(torch.float32).div(255.0).reshape(dims[0], 1, dims[1], dims[2]))
    ys = torch.frombuffer(bytearray(ldata), dtype=torch.uint8).to(torch.int64)
    return xs, ys


def fixture(prefix, n):
    def read(kind):
        name = f"{prefix}-{n}-{kind}-idx{3 if kind == 'images' else 1}-ubyte"
        with open(os.path.join(FIXTURES, name), "rb") as f:
            return f.read()
    return tensors(read("images"), read("labels"))


def archive(prefix):
    root = os.environ["RKTORCH_FASHION_MNIST_DIR"]

    def read(kind):
        name = f"{prefix}-{kind}-idx{3 if kind == 'images' else 1}-ubyte.gz"
        with gzip.open(os.path.join(root, name), "rb") as f:
            return f.read()
    return tensors(read("images"), read("labels"))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(torch.relu(self.bn(self.conv(x))))


def fashion_net():
    return nn.Sequential(ConvBlock(1, 16), ConvBlock(16, 32), nn.Flatten(),
                         nn.Linear(1568, 128), nn.ReLU(), nn.Linear(128, 10))


def decay_split(net):
    named = list(net.named_parameters())
    return ([(n, p) for n, p in named if p.ndim > 1],
            [(n, p) for n, p in named if p.ndim <= 1])


class FashionClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = fashion_net()

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        decay, no_decay = decay_split(self.net)
        opt = torch.optim.AdamW(
            [{"params": [p for _, p in decay]},
             {"params": [p for _, p in no_decay], "weight_decay": 0.0}],
            lr=LR, weight_decay=WEIGHT_DECAY)
        total = self.trainer.estimated_stepping_batches
        warmup = math.ceil(total / 4)
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup),
             torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total - warmup)],
            milestones=[warmup])
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def training_step(self, batch, batch_idx):
        xb, yb = batch
        loss = nn.functional.cross_entropy(self.net(xb), yb)
        self.log("train/loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xb, yb = batch
        logits = self.net(xb)
        self.log("val/loss", nn.functional.cross_entropy(logits, yb))
        self.log("val/acc", logits.argmax(1).eq(yb).float().mean())


class Rows(Logger):
    def __init__(self):
        super().__init__()
        self.rows = []

    @property
    def name(self):
        return "rows"

    @property
    def version(self):
        return 0

    def log_hyperparams(self, params, *args, **kwargs):
        pass

    def log_metrics(self, metrics, step=None):
        self.rows.append([step, {k: float(v) for k, v in metrics.items()}])


class Trace(pl.Callback):
    def __init__(self, steps):
        self.steps = steps
        self.labels, self.losses, self.lrs, self.norms, self.scales = [], [], [], [], []
        self.epochs, self.clock = [], None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.steps:
            return
        self.labels.append(batch[1].tolist())
        self.losses.append(outputs["loss"].item())

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if not self.steps:
            return
        self.lrs.append([g["lr"] for g in optimizer.param_groups])
        params = [p for g in optimizer.param_groups for p in g["params"]]
        self.norms.append(torch.nn.utils.clip_grad_norm_(params, float("inf")).item())
        scaler = getattr(trainer.precision_plugin, "scaler", None)
        if scaler is not None:
            self.scales.append(scaler.get_scale())

    def on_train_epoch_start(self, trainer, pl_module):
        self.clock = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        self.epochs.append([trainer.current_epoch + 1, m["train/loss_epoch"].item(),
                            m["val/loss"].item(), m["val/acc"].item(),
                            time.perf_counter() - self.clock])


class Batched(TensorDataset):
    def __getitems__(self, indices):
        i = torch.tensor(indices, device=self.tensors[0].device)
        return [t[i] for t in self.tensors]


def flat(named):
    return {n: t.detach().cpu().flatten().tolist() for n, t in named}


def jsonable(v):
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    if isinstance(v, list):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    return v


def loaders(mode, device):
    if mode in ("fixture", "golden"):
        (xs, ys), (vx, vy) = fixture("train", 512), fixture("t10k", 128)
        train = DataLoader(TensorDataset(xs, ys), batch_size=BATCH, shuffle=True,
                           generator=torch.Generator().manual_seed(SEED))
        return train, DataLoader(TensorDataset(vx, vy), batch_size=VAL_BATCH)
    target = torch.device(device)
    (xs, ys), (vx, vy) = archive("train"), archive("t10k")
    xs, ys, vx, vy = (t.to(target) for t in (xs, ys, vx, vy))
    train = DataLoader(Batched(xs, ys), batch_size=BATCH, shuffle=True,
                       generator=torch.Generator().manual_seed(SEED), collate_fn=tuple)
    return train, DataLoader(Batched(vx, vy), batch_size=VAL_BATCH, collate_fn=tuple)


def named_state(model):
    return [(n, t.detach().cpu().clone())
            for n, t in [*model.named_parameters(), *model.named_buffers()]]


def fit(device, precision, mode):
    train, val = loaders(mode, device)
    torch.manual_seed(SEED)
    model = FashionClassifier()
    init = named_state(model)
    rows, trace = Rows(), Trace(mode != "full")
    trainer = pl.Trainer(
        accelerator="gpu" if device == "cuda" else "cpu", devices=1, precision=precision,
        max_epochs=EPOCHS, accumulate_grad_batches=ACCUMULATE, gradient_clip_val=CLIP,
        gradient_clip_algorithm="norm",
        log_every_n_steps=50 if mode == "full" else 1, logger=rows, callbacks=[trace],
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        default_root_dir=tempfile.mkdtemp())
    trainer.fit(model, train, val)
    total = trainer.estimated_stepping_batches
    decay, no_decay = decay_split(model.net)
    return {
        "config": {"seed": SEED, "lr": LR, "weight-decay": WEIGHT_DECAY,
                   "accumulate": ACCUMULATE, "clip": CLIP, "batch": BATCH, "epochs": EPOCHS,
                   "total": total, "warmup": math.ceil(total / 4),
                   "lightning": lightning.__version__},
        "names": {"parameters": [n for n, _ in model.named_parameters()],
                  "buffers": [n for n, _ in model.named_buffers()],
                  "decay": ["net." + n for n, _ in decay],
                  "no_decay": ["net." + n for n, _ in no_decay]},
        "init": init,
        "state": named_state(model),
        "labels": trace.labels,
        "losses": trace.losses,
        "lrs": trace.lrs,
        "norms": trace.norms,
        "scales": trace.scales,
        "rows": rows.rows,
        "epochs": trace.epochs,
    }


def summary(t):
    vals = t.flatten().tolist()
    add = math.fsum if t.is_floating_point() else sum
    return {"shape": list(t.shape), "sum": add(vals), "sum-sq": add(v * v for v in vals),
            "head": vals[:8], "tail": vals[-8:]}


def summarized(out):
    keep = ("config", "names", "labels", "losses", "lrs", "norms", "scales", "rows")
    return {**{k: out[k] for k in keep},
            "init": {n: summary(t) for n, t in out["init"]},
            "state": {n: summary(t) for n, t in out["state"]}}


def rktd_atom(v):
    if v is None or v is False:
        return "#f"
    if v is True:
        return "#t"
    if isinstance(v, int):
        return repr(v)
    if isinstance(v, float):
        return repr(v) if math.isfinite(v) else {"inf": "+inf.0", "-inf": "-inf.0"}.get(
            str(v), "+nan.0")
    if isinstance(v, str):
        return json.dumps(v)
    return None


def rktd(v, indent=0):
    atom = rktd_atom(v)
    if atom is not None:
        return atom
    if isinstance(v, list) and all(rktd_atom(x) is not None for x in v):
        return "(" + " ".join(rktd_atom(x) for x in v) + ")"
    flat = rktd_flat(v)
    if indent + len(flat) <= 96:
        return flat
    if isinstance(v, dict):
        return "#hasheq(" + ("\n" + " " * (indent + 8)).join(
            rktd_entry(k, x, indent + 8) for k, x in v.items()) + ")"
    return "(" + ("\n" + " " * (indent + 1)).join(rktd(x, indent + 1) for x in v) + ")"


def rktd_entry(k, v, indent):
    flat = f"({k} . {rktd_flat(v)})"
    if rktd_atom(v) is not None or indent + len(flat) <= 96:
        return flat
    return f"({k} .\n{' ' * (indent + 1)}{rktd(v, indent + 1)})"


def rktd_flat(v):
    atom = rktd_atom(v)
    if atom is not None:
        return atom
    if isinstance(v, dict):
        return "#hasheq(" + " ".join(f"({k} . {rktd_flat(x)})" for k, x in v.items()) + ")"
    return "(" + " ".join(rktd_flat(x) for x in v) + ")"


def golden(path):
    torch.set_num_threads(1)
    data = {"regenerate": "RKTORCH_TWIN_MODE=golden python3 examples/python/20_fashion_mnist.py",
            "lightning": lightning.__version__,
            "torch": torch.__version__.split("+")[0],
            "configs": {p: summarized(fit("cpu", p, "golden")) for p in GOLDEN_PRECISIONS}}
    with open(path, "w") as f:
        f.write(rktd(data) + "\n")


def main():
    warnings.filterwarnings("ignore")
    if MODE == "golden":
        golden(sys.argv[1] if len(sys.argv) > 1 else GOLDEN)
        return
    out = fit(DEVICE, PRECISION, MODE)
    out["init"] = flat(out["init"])
    out["state"] = flat(out["state"])
    out["versions"] = {"python": platform.python_version(), "torch": torch.__version__,
                       "lightning": lightning.__version__, "cuda": torch.version.cuda or "none"}
    if MODE != "full":
        del out["epochs"]
    print(json.dumps(jsonable(out), allow_nan=False))


if __name__ == "__main__":
    main()
