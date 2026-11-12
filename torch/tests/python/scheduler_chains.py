"""torch.optim.lr_scheduler's rates over two parameter groups (lr 0.1 and
0.02): get_last_lr() at construction and after each of twelve
optimizer-then-scheduler steps, for each shape the Racket schedulers
implement, chains, sequences and warm restarts, including sequences that
start a warm restart or a lambda at a milestone, ones that start on a chain
or on a sequence, and one with unsorted milestones (G2-L1). Under "groups",
the seven single shapes' _get_closed_form_lr() at each of those steps, or
get_last_lr() where torch has no closed form (G1-L10). The Racket twin is scheduler-parity-test.rkt.
"""
import json
import warnings

import torch

warnings.simplefilter("ignore")

S = torch.optim.lr_scheduler


def two_groups():
    groups = [{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 0.1},
              {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 0.02}]
    return torch.optim.SGD(groups, lr=0.1)


def stepped(make, read):
    opt = two_groups()
    sched = make(opt)
    out = [read(sched)]
    for _ in range(12):
        opt.step()
        sched.step()
        out.append(read(sched))
    return out


def last(sched):
    return sched.get_last_lr()


def closed(sched):
    if hasattr(sched, "_get_closed_form_lr"):
        return sched._get_closed_form_lr()
    return sched.get_last_lr()


shapes = {
    "step": lambda o: S.StepLR(o, step_size=3, gamma=0.5),
    "multi_step": lambda o: S.MultiStepLR(o, milestones=[2, 5, 9], gamma=0.1),
    "exponential": lambda o: S.ExponentialLR(o, gamma=0.9),
    "cosine": lambda o: S.CosineAnnealingLR(o, T_max=10, eta_min=0.01),
    "linear": lambda o: S.LinearLR(o, start_factor=0.25, end_factor=1.0,
                                   total_iters=4),
    "one_cycle": lambda o: S.OneCycleLR(o, max_lr=1.0, total_steps=12,
                                        cycle_momentum=False),
    "lambda": lambda o: S.LambdaLR(o, lambda t: 1.0 / (t + 1)),
}

composites = {
    "chained_exponential_step": lambda o: S.ChainedScheduler(
        [S.ExponentialLR(o, gamma=0.9), S.StepLR(o, step_size=3, gamma=0.5)]),
    "chained_linear_cosine": lambda o: S.ChainedScheduler(
        [S.LinearLR(o, start_factor=0.25, end_factor=1.0, total_iters=4),
         S.CosineAnnealingLR(o, T_max=10, eta_min=0.01)]),
    "sequential_linear_cosine": lambda o: S.SequentialLR(
        o, [S.LinearLR(o, start_factor=0.25, end_factor=1.0, total_iters=4),
            S.CosineAnnealingLR(o, T_max=8, eta_min=0.01)], milestones=[4]),
    "sequential_exponential_step": lambda o: S.SequentialLR(
        o, [S.ExponentialLR(o, gamma=0.9), S.StepLR(o, step_size=2, gamma=0.5)],
        milestones=[3]),
    "sequential_exponential_warm_restarts": lambda o: S.SequentialLR(
        o, [S.ExponentialLR(o, gamma=0.9),
            S.CosineAnnealingWarmRestarts(o, T_0=2, eta_min=0.01)], milestones=[3]),
    "sequential_chained_linear": lambda o: S.SequentialLR(
        o, [S.ChainedScheduler([S.ExponentialLR(o, gamma=0.9),
                                S.StepLR(o, step_size=3, gamma=0.5)]),
            S.LinearLR(o, start_factor=0.5, end_factor=1.0, total_iters=3)],
        milestones=[5]),
    "sequential_step_lambda": lambda o: S.SequentialLR(
        o, [S.StepLR(o, step_size=2, gamma=0.5),
            S.LambdaLR(o, lambda t: 1.0 / (t + 1))], milestones=[4]),
    "sequential_nested": lambda o: S.SequentialLR(
        o, [S.SequentialLR(o, [S.ExponentialLR(o, gamma=0.9)], milestones=[]),
            S.StepLR(o, step_size=2, gamma=0.5)], milestones=[3]),
    "sequential_unsorted_milestones": lambda o: S.SequentialLR(
        o, [S.ExponentialLR(o, gamma=0.5), S.StepLR(o, step_size=100),
            S.LinearLR(o, start_factor=0.5, total_iters=2)], milestones=[4, 2]),
    "warm_restarts_3_2": lambda o: S.CosineAnnealingWarmRestarts(
        o, T_0=3, T_mult=2, eta_min=0.01),
    "warm_restarts_4_1": lambda o: S.CosineAnnealingWarmRestarts(
        o, T_0=4, T_mult=1, eta_min=0),
}

out = {name: stepped(make, last) for name, make in {**shapes, **composites}.items()}
out["groups"] = {name: stepped(make, closed) for name, make in shapes.items()}
print(json.dumps(out))
