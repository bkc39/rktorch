"""Twin of grad-scaler-parity-test.rkt (G3-L4 of the T2 spec, #385).

torch.amp.GradScaler(device="cpu", init_scale=1024, growth_interval=2) over a
scripted eight-step run of SGD with momentum on one two-element parameter;
the loss is the parameter dotted with a coefficient pair that is infinite at
steps 2 and 5. Prints the scale after each update and the final parameter.
"""

import json

import torch

INF_STEPS = (2, 5)


def coefficients(t):
    first = float("inf") if t in INF_STEPS else 0.5 + 0.1 * t
    return [first, 0.25 - 0.05 * t]


p = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
opt = torch.optim.SGD([p], lr=0.1, momentum=0.9)
scaler = torch.amp.GradScaler(device="cpu", init_scale=1024.0, growth_interval=2)
scales = []
for t in range(8):
    opt.zero_grad(set_to_none=True)
    scaler.scale((p * torch.tensor(coefficients(t))).sum()).backward()
    scaler.step(opt)
    scaler.update()
    scales.append(scaler.get_scale())

print(json.dumps({"scales": scales, "params": p.detach().tolist()}))
