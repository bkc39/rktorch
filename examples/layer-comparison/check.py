import json
from pathlib import Path

import torch
from models import SmallResNet, TransformerStack


def tensor(record):
    return torch.tensor(record["data"], dtype=torch.float32).reshape(record["shape"])


for record in json.loads(Path(__file__).with_name("racket-reference.json").read_text()):
    model = SmallResNet() if record["name"] == "resnet" else TransformerStack(32, 4, 2, 16)
    params = dict(model.named_parameters())
    assert list(params) == [p["name"].replace("-", "_") for p in record["parameters"]]
    with torch.no_grad():
        for p in record["parameters"]:
            params[p["name"].replace("-", "_")].copy_(tensor(p["tensor"]))
    model.eval()
    x = tensor(record["input"])
    output = model(x)
    reference = tensor(record["output"])
    torch.testing.assert_close(output, reference, atol=2e-5, rtol=2e-4)
    target = torch.arange(output.numel(), dtype=torch.float32).reshape(output.shape) * 0.001
    (output * target).mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
    if record["name"] == "transformer":
        changed = x.clone()
        changed[:, 4:] += 100
        torch.testing.assert_close(model(changed)[:, :4], output[:, :4], atol=0, rtol=0)
        assert len(list(model.buffers())) == 2
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    model.train()
    (model(x) * target).mean().backward()
    optimizer.step()
    print(record["name"], "Racket/Python output parity, parameter names, gradients, Adam OK;",
          "max error", (output - reference).abs().max().item())

records = json.loads(Path(__file__).with_name("racket-reference.json").read_text())
with Path(__file__).with_name("ocaml-reference.txt").open("w") as out:
    def write_tensor(record):
        out.write(f"{len(record['shape'])} " + " ".join(map(str, record["shape"])) + "\n")
        out.write(" ".join(map(str, record["data"])) + "\n")
    out.write(f"{len(records)}\n")
    for record in records:
        out.write(record["name"] + "\n")
        write_tensor(record["input"])
        write_tensor(record["output"])
        out.write(f"{len(record['parameters'])}\n")
        for parameter in record["parameters"]:
            out.write(parameter["name"].replace("-", "_") + "\n")
            write_tensor(parameter["tensor"])
