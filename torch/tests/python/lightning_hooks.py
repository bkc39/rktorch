"""Lightning 2.6.5's hook order, the oracle of trainer-hook-order-test.rkt.

Records every hook a LightningModule and a Callback can override, in the order
Lightning calls them, for the scenarios LO-1..LO-8 of the T0 spec (#383)
and for fits that skip a window by a None loss or an optimizer_step that does
not step (#446), dumps _FxValidator.functions, and records the step each
logger row is labelled with in fits that skip a window (#445), and records
trainer.estimated_stepping_batches as configure_optimizers reads it for a
table of fit options (G2-L11 of the T2 spec, #385). Writes
torch/tests/lightning-hooks.rktd, or the path given as the first argument;
lightning-hooks-parity-test.rkt regenerates it and compares.

Each event is (name epoch global-step batch-idx lr): the raw Lightning name
(Callback.<hook> for the callback), trainer.current_epoch,
trainer.global_step, the hook's own batch_idx argument (#f when it takes
none) and the first parameter group's learning rate (#f before
configure_optimizers and outside fit). Each logged row is (step (key ...)):
the step log_metrics received and the row's keys, sorted.
"""

import inspect
import os
import sys

import lightning
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.core.hooks import CheckpointHooks, DataHooks, ModelHooks
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.trainer.connectors.logger_connector.fx_validator import (
    _FxValidator,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "lightning-hooks.rktd")

MODULE_METHODS = [
    "training_step", "validation_step", "test_step", "predict_step",
    "configure_optimizers", "configure_callbacks", "optimizer_step",
    "optimizer_zero_grad", "backward", "lr_scheduler_step",
    "configure_gradient_clipping", "clip_gradients", "forward",
]


def public_hooks(*classes):
    names = []
    for cls in classes:
        for name, value in vars(cls).items():
            if (inspect.isfunction(value) and not name.startswith("_")
                    and not name.endswith("_dataloader")
                    and name != "configure_sharded_model"
                    and name not in names):
                names.append(name)
    return names


MODULE_HOOKS = public_hooks(ModelHooks, DataHooks, CheckpointHooks) + [
    m for m in MODULE_METHODS
    if m not in public_hooks(ModelHooks, DataHooks, CheckpointHooks)]
CALLBACK_HOOKS = public_hooks(Callback)


class Scenario:
    def __init__(self, skip_epoch=None, fail_at=None, stop_at=None,
                 step_scheduler=False, skip_loss=None, skip_step=None):
        self.skip_epoch = skip_epoch
        self.skip_loss = skip_loss
        self.skip_step = skip_step
        self.fail_at = fail_at
        self.stop_at = stop_at
        self.step_scheduler = step_scheduler
        self.trainer = None
        self.events = []


class Failure(Exception):
    pass


class Symbol(str):
    pass


def current_lr(trainer):
    if not trainer.optimizers:
        return None
    return trainer.optimizers[0].param_groups[0]["lr"]


def record(scenario, name, base, args, kwargs):
    bound = inspect.signature(base).bind(None, *args, **kwargs).arguments
    trainer = scenario.trainer
    scenario.events.append((Symbol(name), trainer.current_epoch, trainer.global_step,
                            bound.get("batch_idx"), current_lr(trainer)))


class Base(pl.LightningModule):
    def __init__(self, scenario):
        super().__init__()
        self.scenario = scenario
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        return self.layer(x)

    def at(self, place, batch_idx):
        here = (self.current_epoch, batch_idx)
        return place == "every" or place == here or (
            isinstance(place, list) and here in place)

    def training_step(self, batch, batch_idx):
        if self.at(self.scenario.fail_at, batch_idx):
            raise Failure("training-step")
        if self.at(self.scenario.stop_at, batch_idx):
            self.trainer.should_stop = True
        if self.at(self.scenario.skip_loss, batch_idx):
            return None
        x, y = batch
        return nn.functional.mse_loss(self(x), y)

    def optimizer_step(self, epoch, batch_idx, optimizer,
                       optimizer_closure=None):
        if self.at(self.scenario.skip_step, batch_idx):
            optimizer_closure()
        else:
            optimizer.step(closure=optimizer_closure)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        return nn.functional.mse_loss(self(x), y)

    def test_step(self, batch, batch_idx):
        x, y = batch
        return nn.functional.mse_loss(self(x), y)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)

    def on_train_batch_start(self, batch, batch_idx):
        if self.at(self.scenario.skip_epoch, batch_idx):
            return -1
        return None

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        if self.scenario.step_scheduler:
            return {"optimizer": opt,
                    "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        return {"optimizer": opt, "lr_scheduler": sched}


def recording(cls, names, prefix):
    def wrap(name):
        base = getattr(cls, name)

        def hook(self, *args, **kwargs):
            record(self.scenario, prefix + name, base, args, kwargs)
            return base(self, *args, **kwargs)
        return hook
    return type("Recording" + cls.__name__, (cls,),
                {name: wrap(name) for name in names})


class RecorderCallback(Callback):
    def __init__(self, scenario):
        self.scenario = scenario


Recorder = recording(Base, MODULE_HOOKS, "")
Recorded = recording(RecorderCallback, CALLBACK_HOOKS, "Callback.")


def loader(n):
    g = torch.Generator().manual_seed(n)
    xs = torch.randn(n, 2, generator=g)
    return DataLoader(TensorDataset(xs, xs.sum(1, keepdim=True)), batch_size=2)


def trainer(callbacks, logger=False, **kwargs):
    options = {"max_epochs": 2, "num_sanity_val_steps": 1}
    options.update(kwargs)
    return pl.Trainer(enable_checkpointing=False, logger=logger,
                      enable_progress_bar=False, enable_model_summary=False,
                      accelerator="cpu", devices=1,
                      callbacks=callbacks, **options)


def run(entry, scenario_kwargs=None, train=6, trainer_kwargs=None):
    torch.manual_seed(0)
    scenario = Scenario(**(scenario_kwargs or {}))
    model = Recorder(scenario)
    t = trainer([Recorded(scenario)], **(trainer_kwargs or {}))
    scenario.trainer = t
    try:
        if entry == "fit":
            t.fit(model, loader(train), loader(4))
        elif entry == "test":
            t.test(model, loader(4), verbose=False)
        else:
            t.predict(model, loader(4))
    except Failure:
        pass
    return scenario.events


SCENARIOS = {
    "fit-val": lambda: run("fit"),
    "accumulate": lambda: run(
        "fit", {"step_scheduler": True}, train=10,
        trainer_kwargs={"accumulate_grad_batches": 2}),
    "skip-epoch": lambda: run("fit", {"skip_epoch": (0, 1)}),
    "exception": lambda: run("fit", {"fail_at": (0, 1)}),
    "request-stop": lambda: run("fit", {"stop_at": (0, 1)}),
    "val-check-interval": lambda: run(
        "fit", train=10, trainer_kwargs={"val_check_interval": 2}),
    "skip-loss": lambda: run("fit", {"skip_loss": (0, 1)}),
    "skip-step": lambda: run("fit", {"skip_step": (0, 1)}),
    "skip-loss-accumulate": lambda: run(
        "fit", {"skip_loss": (0, 0), "step_scheduler": True}, train=10,
        trainer_kwargs={"accumulate_grad_batches": 2}),
    "skip-loss-end-accumulate": lambda: run(
        "fit", {"skip_loss": (0, 1), "step_scheduler": True}, train=10,
        trainer_kwargs={"accumulate_grad_batches": 2}),
    "skip-loss-last-accumulate": lambda: run(
        "fit", {"skip_loss": (0, 4), "step_scheduler": True}, train=10,
        trainer_kwargs={"accumulate_grad_batches": 2}),
    "skip-window-accumulate": lambda: run(
        "fit", {"skip_loss": [(0, 0), (0, 1)], "step_scheduler": True},
        train=10, trainer_kwargs={"accumulate_grad_batches": 2}),
    "skip-step-accumulate": lambda: run(
        "fit", {"skip_step": (0, 1), "step_scheduler": True}, train=10,
        trainer_kwargs={"accumulate_grad_batches": 2}),
    "skip-every-max-steps": lambda: run(
        "fit", {"skip_loss": "every"},
        trainer_kwargs={"max_epochs": -1, "max_steps": 4}),
    "test": lambda: run("test"),
    "predict": lambda: run("predict"),
}


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

    def log_hyperparams(self, *args, **kwargs):
        pass

    def log_metrics(self, metrics, step=None):
        self.rows.append((step, [Symbol(k) for k in sorted(metrics)]))


class Logged(Base):
    def training_step(self, batch, batch_idx):
        self.log("tl", 1.0, on_step=True, on_epoch=True)
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self.log("vl", 2.0)
        return super().validation_step(batch, batch_idx)


def logged(scenario_kwargs, train=6, accumulate=1, log_every=1):
    torch.manual_seed(0)
    rows = Rows()
    t = trainer([], logger=rows, log_every_n_steps=log_every,
                accumulate_grad_batches=accumulate)
    t.fit(Logged(Scenario(**scenario_kwargs)), loader(train), loader(4))
    return rows.rows


LOGGED = {
    "skip-loss": lambda: logged({"skip_loss": (0, 1)}),
    "skip-step": lambda: logged({"skip_step": (0, 1)}),
    "skip-loss-accumulate": lambda: logged(
        {"skip_loss": (0, 1)}, train=10, accumulate=2),
    "skip-loss-log-every-2": lambda: logged(
        {"skip_loss": (0, 1)}, log_every=2),
}


class Stepping(Base):
    def configure_optimizers(self):
        self.scenario.events.append(self.trainer.estimated_stepping_batches)
        return torch.optim.SGD(self.parameters(), lr=0.1)


STEPPING = [
    {"train": 5, "accumulate": 1, "limit": 1.0, "max-epochs": 2},
    {"train": 5, "accumulate": 2, "limit": 1.0, "max-epochs": 3},
    {"train": 5, "accumulate": 3, "limit": 1.0, "max-epochs": 2,
     "max-steps": 3},
    {"train": 6, "accumulate": 1, "limit": 0.5, "max-epochs": 2},
    {"train": 6, "accumulate": 4, "limit": 4, "max-epochs": 1},
    {"train": 3, "accumulate": 5, "limit": 1.0, "max-epochs": 2},
    {"train": 4, "accumulate": 1, "limit": 1.0, "max-steps": 7},
    {"train": 5, "accumulate": 2, "limit": 1.0, "fast-dev-run": 3},
    {"train": 2, "accumulate": 1, "limit": 1.0, "fast-dev-run": 5},
    {"train": 4, "accumulate": 1, "limit": 1.0, "fast-dev-run": True},
]


def estimated(row):
    torch.manual_seed(0)
    scenario = Scenario()
    t = trainer([], max_epochs=row.get("max-epochs"),
                max_steps=row.get("max-steps", -1),
                accumulate_grad_batches=row["accumulate"],
                limit_train_batches=row["limit"],
                fast_dev_run=row.get("fast-dev-run", False))
    t.fit(Stepping(scenario), loader(2 * row["train"]))
    [steps] = scenario.events
    return {**row, "estimated": steps}


def rktd(v):
    if v is None or v is False:
        return "#f"
    if v is True:
        return "#t"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, Symbol):
        return str(v)
    if isinstance(v, str):
        return '"' + v + '"'
    return "(" + " ".join(rktd(x) for x in v) + ")"


def permissions():
    out = []
    for name in sorted(_FxValidator.functions):
        opts = _FxValidator.functions[name]
        value = "#f" if opts is None else rktd(
            (opts["allowed_on_step"], opts["allowed_on_epoch"],
             opts["default_on_step"], opts["default_on_epoch"]))
        out.append("   (%s . %s)" % (name, value))
    return "\n".join(out)


def stepping_text():
    rows = []
    for row in STEPPING:
        fields = " ".join("(%s . %s)" % (k, rktd(v))
                          for k, v in estimated(row).items())
        rows.append("     #hasheq(" + fields + ")")
    return "\n".join(rows)


def scenario_text(name, events):
    rows = "\n".join("     " + rktd(e) for e in events)
    return "   (%s\n    .\n    (\n%s))" % (name, rows)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else OUT
    scenarios = "\n".join(scenario_text(name, SCENARIOS[name]())
                          for name in sorted(SCENARIOS))
    rows = "\n".join(scenario_text(name, LOGGED[name]())
                     for name in sorted(LOGGED))
    text = ("#hasheq((lightning . %s)\n"
            "        (torch . %s)\n"
            "        (log-permissions\n         .\n         #hasheq(\n%s))\n"
            "        (logged-steps\n         .\n         #hasheq(\n%s))\n"
            "        (scenarios\n         .\n         #hasheq(\n%s))\n"
            "        (stepping\n         .\n         (\n%s)))\n") % (
        rktd(lightning.__version__), rktd(torch.__version__.split("+")[0]),
        permissions(), rows, scenarios, stepping_text())
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
