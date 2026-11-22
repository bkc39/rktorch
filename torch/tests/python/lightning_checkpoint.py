"""Lightning 2.6.5's ModelCheckpoint over a scripted val-loss list, the oracle
of model-checkpoint-test.rkt (T3-L25 of the T3 spec, #386).

Each case fits a two-parameter module whose validation step logs the epoch's
entry of SCORES as "val-loss", and records every file the callback saves or
deletes, in order, then its best_k_models, best and k-th scores, last path
and state key. Paths are recorded by file name, so the fixture holds no
directory. Writes torch/tests/lightning-checkpoint.rktd, or the path given as
the first argument; model-checkpoint-parity-test.rkt regenerates it and
compares.
"""

import math
import os
import sys
import tempfile

import lightning
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "lightning-checkpoint.rktd")

SCORES = [0.5, 0.3, 0.4, float("nan"), 0.2, 0.6]


class Scripted(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return nn.functional.mse_loss(self.layer(x), y)

    def validation_step(self, batch, batch_idx):
        self.log("val-loss", SCORES[self.current_epoch])

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def loader(n):
    g = torch.Generator().manual_seed(n)
    xs = torch.randn(n, 2, generator=g)
    return DataLoader(TensorDataset(xs, xs.sum(1, keepdim=True)), batch_size=2)


class Symbol(str):
    pass


class Pair(tuple):
    def __new__(cls, a, d):
        return super().__new__(cls, (a, d))


def recorded(callback, events):
    save, remove = callback._save_checkpoint, callback._remove_checkpoint

    def save_checkpoint(trainer, filepath):
        events.append((Symbol("save"), os.path.basename(filepath)))
        save(trainer, filepath)

    def remove_checkpoint(trainer, filepath):
        events.append((Symbol("delete"), os.path.basename(filepath)))
        remove(trainer, filepath)

    callback._save_checkpoint = save_checkpoint
    callback._remove_checkpoint = remove_checkpoint
    return callback


def score(v):
    return None if v is None else float(v)


def named(path):
    return os.path.basename(path) if path else None


def fitted(root, name, existing=(), accumulate=1, **checkpoint_kwargs):
    dirpath = os.path.join(root, name)
    os.makedirs(dirpath)
    for f in existing:
        open(os.path.join(dirpath, f), "w").close()
    events = []
    callback = recorded(ModelCheckpoint(dirpath=dirpath, **checkpoint_kwargs), events)
    torch.manual_seed(0)
    pl.Trainer(max_epochs=len(SCORES), num_sanity_val_steps=0, logger=False,
               enable_progress_bar=False, enable_model_summary=False,
               accelerator="cpu", devices=1, accumulate_grad_batches=accumulate,
               callbacks=[callback]).fit(Scripted(), loader(6), loader(2))
    best_k = sorted((named(p), score(v)) for p, v in callback.best_k_models.items())
    return events + [
        (Symbol("best-k"), *[Pair(k, v) for k, v in best_k]),
        (Symbol("best"), named(callback.best_model_path), score(callback.best_model_score)),
        (Symbol("kth"), named(callback.kth_best_model_path), score(callback.kth_value)),
        Pair(Symbol("last"), named(callback.last_model_path)),
        Pair(Symbol("state-key"), callback.state_key),
    ]


CASES = {
    "no-monitor": lambda root: fitted(root, "no-monitor"),
    "min-top-2": lambda root: fitted(root, "min-top-2", monitor="val-loss", save_top_k=2,
                                     filename="{epoch}-{val-loss:.2f}"),
    "max-top-all": lambda root: fitted(root, "max-top-all", monitor="val-loss",
                                       save_top_k=-1, mode="max"),
    "save-last": lambda root: fitted(root, "save-last", monitor="val-loss", save_last=True),
    "pre-existing": lambda root: fitted(root, "pre-existing", existing=["epoch=0-step=3.ckpt"]),
    "unformatted": lambda root: fitted(root, "unformatted", monitor="val-loss",
                                       filename="{epoch}-{val-loss}"),
    "every-2-steps": lambda root: fitted(root, "every-2-steps", accumulate=2,
                                         every_n_train_steps=2, save_top_k=-1),
}


def rktd(v):
    if v is None or v is False:
        return "#f"
    if v is True:
        return "#t"
    if isinstance(v, int):
        return repr(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "+nan.0"
        if math.isinf(v):
            return "+inf.0" if v > 0 else "-inf.0"
        return repr(v)
    if isinstance(v, Symbol):
        return str(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, Pair):
        return "(" + rktd(v[0]) + " . " + rktd(v[1]) + ")"
    return "(" + " ".join(rktd(x) for x in v) + ")"


def case_text(name, events):
    rows = "\n".join("     " + rktd(e) for e in events)
    return "   (%s\n    .\n    (\n%s))" % (name, rows)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else OUT
    with tempfile.TemporaryDirectory() as root:
        cases = "\n".join(case_text(name, CASES[name](root)) for name in sorted(CASES))
    text = ("#hasheq((lightning . %s)\n"
            "        (cases\n         .\n         #hasheq(\n%s)))\n") % (
        rktd(lightning.__version__), cases)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
