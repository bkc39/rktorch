"""torch.optim.lr_scheduler's rates over twelve steps for each shape the
Racket schedulers implement: the rate at construction and after each of
eleven optimizer-then-scheduler steps.
"""
import json
import torch


def rates(make):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=0.1)
    sched = make(opt)
    out = [sched.get_last_lr()[0]]
    for _ in range(11):
        opt.step()
        sched.step()
        out.append(sched.get_last_lr()[0])
    return out


S = torch.optim.lr_scheduler
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
print(json.dumps({name: rates(make) for name, make in shapes.items()}))
