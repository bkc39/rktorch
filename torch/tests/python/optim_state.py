"""torch's optimizer, scheduler and grad-scaler state_dicts, the oracles of
optim-state-dict-parity-test.rkt, scheduler-state-parity-test.rkt and
grad-scaler-state-parity-test.rkt (T3-L29..L31 of the T3 spec, #386).

For each optimizer case, writes <case>.safetensors and <case>.json into the
directory given as the first argument. The safetensors file holds a
Linear(3, 2)'s initial parameters, five fixed gradients, the parameters after
three and after five steps, and the state tensors after three steps; the JSON
holds the constructor's groups, the momentum some cases set after a given step,
and torch's state_dict after three steps, with
parameter indices mapped to names in named_parameters order and every state
tensor replaced by the name of its safetensors entry. Each scheduler case
writes <case>.json: the state_dict after every step. The grad-scaler case
writes grad-scaler.json on the CPU, or on RKTORCH_PARITY_DEVICE.
"""

import json
import os
import struct
import sys

import torch
from torch import nn

TAGS = {torch.float32: "F32", torch.float64: "F64", torch.int64: "I64"}

STEPS_BEFORE, STEPS_AFTER = 3, 2


def write_safetensors(path, entries):
    header, payload = {}, b""
    for name, t in entries.items():
        raw = t.detach().contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": TAGS[t.dtype], "shape": list(t.shape),
                        "data_offsets": [len(payload), len(payload) + len(raw)]}
        payload += raw
    text = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text + payload)


def model():
    torch.manual_seed(0)
    return nn.Linear(3, 2)


def gradients(net):
    g = torch.Generator().manual_seed(1)
    return [{name: torch.randn(p.shape, generator=g) for name, p in net.named_parameters()}
            for _ in range(STEPS_BEFORE + STEPS_AFTER)]


def stepped(opt, net, grads, start, changes):
    params = dict(net.named_parameters())
    for k, g in enumerate(grads, start + 1):
        for name, p in params.items():
            p.grad = g[name].clone()
        opt.step()
        for group in opt.param_groups if k in changes else []:
            group["momentum"] = changes[k]


def named_state(opt, net, entries):
    names = [name for name, _ in net.named_parameters()]
    sd = opt.state_dict()
    state = {}
    for index, slots in sd["state"].items():
        name = names[index]
        state[name] = {}
        for slot, v in slots.items():
            if torch.is_tensor(v) and slot != "step":
                key = "state.%s.%s" % (name, slot)
                entries[key] = v.clone()
                state[name][slot] = key
            else:
                state[name][slot] = int(v)
    groups = [{**g, "params": [names[i] for i in g["params"]]} for g in sd["param_groups"]]
    return {"state": state, "param_groups": groups}


def groups_of(net, split):
    if not split:
        return [{"params": ["weight", "bias"]}]
    return [{"params": ["weight"], **split[0]}, {"params": ["bias"], **split[1]}]


def optimizer_case(out, case, make, kwargs, split=None, changes=None):
    changes = changes or {}
    net = model()
    entries = {"initial." + n: p.detach().clone() for n, p in net.named_parameters()}
    grads = gradients(net)
    for k, g in enumerate(grads):
        entries.update({"grad.%d.%s" % (k, n): t for n, t in g.items()})
    params = dict(net.named_parameters())
    groups = groups_of(net, split)
    opt = make([{**g, "params": [params[n] for n in g["params"]]} for g in groups], **kwargs)
    stepped(opt, net, grads[:STEPS_BEFORE], 0, changes)
    entries.update({"after3." + n: p.detach().clone() for n, p in params.items()})
    state = named_state(opt, net, entries)
    stepped(opt, net, grads[STEPS_BEFORE:], STEPS_BEFORE, changes)
    entries.update({"after5." + n: p.detach().clone() for n, p in params.items()})
    write_safetensors(os.path.join(out, case + ".safetensors"), entries)
    with open(os.path.join(out, case + ".json"), "w") as f:
        json.dump({"optimizer": make.__name__.lower(), "kwargs": kwargs, "groups": groups,
                   "momentum_after": [[k, m] for k, m in sorted(changes.items())],
                   "state_dict": state}, f, indent=1)


OPTIMIZERS = {
    "sgd": (torch.optim.SGD, {"lr": 0.1}),
    "sgd-momentum": (torch.optim.SGD, {"lr": 0.1, "momentum": 0.9, "weight_decay": 0.01}),
    "sgd-nesterov": (torch.optim.SGD, {"lr": 0.1, "momentum": 0.9, "nesterov": True}),
    "adam": (torch.optim.Adam, {"lr": 0.01, "betas": (0.9, 0.99), "weight_decay": 0.01}),
    "adamw": (torch.optim.AdamW, {"lr": 0.01},
              [{"weight_decay": 0.1}, {"lr": 0.02, "weight_decay": 0.0}]),
    "rmsprop": (torch.optim.RMSprop, {"lr": 0.01, "alpha": 0.9}),
    "rmsprop-momentum": (torch.optim.RMSprop, {"lr": 0.01, "alpha": 0.9, "momentum": 0.9}),
    "sgd-momentum-fall": (torch.optim.SGD, {"lr": 0.1, "momentum": 0.9}, None, {2: 0.0, 4: 0.9}),
    "sgd-momentum-rise": (torch.optim.SGD, {"lr": 0.1}, None, {3: 0.9}),
    "rmsprop-momentum-fall": (torch.optim.RMSprop, {"lr": 0.01, "alpha": 0.9, "momentum": 0.9},
                              None, {2: 0.0, 4: 0.9}),
}

S = torch.optim.lr_scheduler

SCHEDULERS = {
    "step": lambda o: S.StepLR(o, step_size=2, gamma=0.5),
    "multi-step": lambda o: S.MultiStepLR(o, milestones=[2, 4], gamma=0.5),
    "exponential": lambda o: S.ExponentialLR(o, gamma=0.9),
    "cosine": lambda o: S.CosineAnnealingLR(o, T_max=5, eta_min=0.001),
    "linear": lambda o: S.LinearLR(o, start_factor=0.5, total_iters=4),
    "one-cycle": lambda o: S.OneCycleLR(o, max_lr=0.1, total_steps=10),
    "lambda": lambda o: S.LambdaLR(o, lambda t: 0.95 ** t),
    "chained": lambda o: S.ChainedScheduler(
        [S.StepLR(o, step_size=2, gamma=0.5), S.ExponentialLR(o, gamma=0.9)]),
    "sequential": lambda o: S.SequentialLR(
        o, [S.LinearLR(o, start_factor=0.5, total_iters=2), S.CosineAnnealingLR(o, T_max=5)],
        milestones=[2]),
    "warm-restarts": lambda o: S.CosineAnnealingWarmRestarts(o, T_0=3, T_mult=2),
}

SCHEDULER_KEYS = ("last_epoch", "base_lrs", "_last_lr", "T_cur", "T_i")


def scheduler_state(s):
    sd = s.state_dict()
    out = {k: sd[k] for k in SCHEDULER_KEYS if k in sd}
    inner = getattr(s, "_schedulers", None)
    if inner is not None:
        out["_schedulers"] = [scheduler_state(x) for x in inner]
    return out


def scheduler_case(out, case, make, steps=7):
    net = model()
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    s = make(opt)
    states = [scheduler_state(s)]
    for _ in range(steps):
        opt.step()
        s.step()
        states.append(scheduler_state(s))
    with open(os.path.join(out, case + ".json"), "w") as f:
        json.dump({"states": states}, f, indent=1)


def grad_scaler_case(out, device):
    net = model().to(device)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler(device, init_scale=2.0 ** 4, growth_interval=2)
    states = []
    for k, overflow in enumerate([False, True, False, False, False, True]):
        loss = net(torch.ones(1, 3, device=device)).sum()
        scaler.scale(loss * (float("inf") if overflow else 1.0)).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        states.append({"overflow": overflow, **scaler.state_dict()})
    with open(os.path.join(out, "grad-scaler.json"), "w") as f:
        json.dump({"device": device, "states": states}, f, indent=1)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for case, spec in OPTIMIZERS.items():
        optimizer_case(out, case, *spec)
    for case, make in SCHEDULERS.items():
        scheduler_case(out, case, make)
    grad_scaler_case(out, os.environ.get("RKTORCH_PARITY_DEVICE", "cpu"))


if __name__ == "__main__":
    main()
