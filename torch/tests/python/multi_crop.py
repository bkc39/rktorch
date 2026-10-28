"""The multi-crop of examples/python/18_dino.py, for augment-parity-test.rkt.

The first 16 CIFAR-10 fixture records, as uint8 on RKTORCH_PARITY_DEVICE,
go through the twin's multi_crop under torch.Generator(device) seeded 0 and
1, with its CIFAR-10 mean and std.
"""
import importlib.util
import json
import os

import torch

from pretrained_common import packed

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "..", "..", "..", "examples", "python")
FIXTURE = os.path.join(HERE, "..", "..", "vision", "fixtures",
                       "cifar10-256.bin")
RECORD = 3073
IMAGES = 16

device = os.environ.get("RKTORCH_PARITY_DEVICE", "cpu")


def dino():
    spec = importlib.util.spec_from_file_location(
        "dino", os.path.join(EXAMPLES, "18_dino.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


with open(FIXTURE, "rb") as f:
    data = f.read(RECORD * IMAGES)
pixels = b"".join(data[i * RECORD + 1:(i + 1) * RECORD] for i in range(IMAGES))
images = (torch.frombuffer(bytearray(pixels), dtype=torch.uint8)
          .view(IMAGES, 3, 32, 32).to(device))

twin = dino()
mean = torch.tensor(twin.MEAN, device=device).view(1, 3, 1, 1)
std = torch.tensor(twin.STD, device=device).view(1, 3, 1, 1)
out = {}
for seed in (0, 1):
    gen = torch.Generator(device).manual_seed(seed)
    big, small = twin.multi_crop(images, gen, mean, std)
    out[str(seed)] = {"global": packed(big.cpu()), "local": packed(small.cpu())}
print(json.dumps(out))
