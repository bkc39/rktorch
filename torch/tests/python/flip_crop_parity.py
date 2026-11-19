"""torchvision's RandomHorizontalFlip and RandomCrop, for flip-crop-parity-test.rkt.

Each case seeds the global stream, then runs its transforms image by image,
as a per-sample pipeline does. The first CIFAR-10 fixture records are the
uint8 images; the float32 ones are the same pixels over 255.
"""
import json
import os

import torch
import torchvision.transforms as v1
from torchvision.transforms import v2

from pretrained_common import packed

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "..", "..", "vision", "fixtures",
                       "cifar10-256.bin")
RECORD = 3073
IMAGES = 12

with open(FIXTURE, "rb") as f:
    data = f.read(RECORD * IMAGES)
pixels = b"".join(data[i * RECORD + 1:(i + 1) * RECORD] for i in range(IMAGES))
cifar = torch.frombuffer(bytearray(pixels), dtype=torch.uint8).view(
    IMAGES, 3, 32, 32)
odd = (torch.arange(5 * 2 * 5 * 7) % 251).to(torch.uint8).view(5, 2, 5, 7)

INPUTS = {"cifar": cifar, "odd": odd}


def transform(api, op, size):
    name, arg = op
    if name == "flip":
        return api.RandomHorizontalFlip(arg)
    return api.RandomCrop(size, padding=arg)


def run(api, ops, images, seed):
    size = tuple(images.shape[-2:])
    chain = api.Compose([transform(api, op, size) for op in ops])
    torch.manual_seed(seed)
    return torch.stack([chain(image) for image in images])


CASES = [
    ("cifar", [["flip", 0.5]], [0, 7, 2026]),
    ("cifar", [["flip", 0.3]], [1]),
    ("cifar", [["flip", 0.0]], [3]),
    ("cifar", [["flip", 1.0]], [3]),
    ("cifar", [["crop", 4]], [0, 7, 2026]),
    ("cifar", [["crop", 0]], [5]),
    ("odd", [["crop", 3]], [11]),
    ("odd", [["flip", 0.5]], [11]),
    ("cifar", [["crop", 4], ["flip", 0.5]], [0, 42]),
    ("odd", [["crop", 2], ["flip", 0.5]], [9]),
]

out = []
for source, ops, seeds in CASES:
    for dtype in ("uint8", "float32"):
        images = INPUTS[source]
        if dtype == "float32":
            images = images.to(torch.float32) / 255
        for seed in seeds:
            want = run(v1, ops, images, seed)
            assert torch.equal(want, run(v2, ops, images, seed)), (ops, seed)
            out.append({"name": f"{source} {ops} {dtype} seed {seed}",
                        "ops": ops, "seed": seed, "dtype": dtype,
                        "input": packed(images), "output": packed(want)})
print(json.dumps(out))
