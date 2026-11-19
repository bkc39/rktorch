"""The committed Fashion-MNIST fixture read by torchvision's FashionMNIST.

Copies the four fixture files into a temporary FashionMNIST/raw/, loads each
split with transform=ToTensor(), and prints the stacked images and labels as
packed little-endian buffers, with torchvision's class names.
"""
import json
import os
import shutil
import tempfile

import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "..", "vision",
                        "fixtures", "fashion-mnist")
FILES = {
    "train-512-images-idx3-ubyte": "train-images-idx3-ubyte",
    "train-512-labels-idx1-ubyte": "train-labels-idx1-ubyte",
    "t10k-128-images-idx3-ubyte": "t10k-images-idx3-ubyte",
    "t10k-128-labels-idx1-ubyte": "t10k-labels-idx1-ubyte",
}


def packed(t):
    return {"hex": t.contiguous().numpy().tobytes().hex(), "shape": list(t.shape)}


def split(root, train):
    ds = FashionMNIST(root, train=train, transform=ToTensor(), download=False)
    images = torch.stack([ds[i][0] for i in range(len(ds))])
    labels = torch.tensor([ds[i][1] for i in range(len(ds))], dtype=torch.int64)
    return {"images": packed(images), "labels": packed(labels)}


with tempfile.TemporaryDirectory() as root:
    raw = os.path.join(root, "FashionMNIST", "raw")
    os.makedirs(raw)
    for ours, theirs in FILES.items():
        shutil.copyfile(os.path.join(FIXTURES, ours), os.path.join(raw, theirs))
    print(json.dumps({"train": split(root, True), "test": split(root, False),
                      "classes": FashionMNIST.classes}))
