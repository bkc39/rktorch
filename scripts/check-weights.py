"""Checks that the checkpoints torch/vision/weights.rkt fetches from
Hugging Face hold torchvision's ImageNet weights, tensor for tensor:

    nix develop --command python3 scripts/check-weights.py WEIGHTS_DIR [NAME ...]

WEIGHTS_DIR is the cache `pretrained-weights` fills. Each file is timm's
`tv_in1k` copy of torchvision's IMAGENET1K_V1 weights with torchvision's
key names (`bn1.running_mean`, `layer1.0.downsample.0.weight`), which
weights.rkt renames on load. The script compares every tensor with
torchvision's own state_dict, dtype and bits, and prints each file's size
and SHA-256, the values weights.rkt records.

`--categories PATH` also writes the ImageNet class names, one per line,
in label order, from the same weights' metadata.
"""
import argparse
import hashlib
import json
import os
import struct
import sys

import torch
from torchvision import models

CHECKPOINTS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
}

DTYPES = {"F32": torch.float32, "I64": torch.int64}


def load_safetensors(raw):
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n])
    header.pop("__metadata__", None)
    body = raw[8 + n:]
    return {k: torch.frombuffer(bytearray(body[v["data_offsets"][0]:
                                               v["data_offsets"][1]]),
                                dtype=DTYPES[v["dtype"]]).reshape(v["shape"])
            for k, v in header.items()}


def differences(ours, theirs):
    for key in sorted(ours.keys() ^ theirs.keys()):
        yield f"{key}: only in {'the file' if key in ours else 'torchvision'}"
    for key in sorted(ours.keys() & theirs.keys()):
        a, b = ours[key], theirs[key]
        if a.dtype != b.dtype or a.shape != b.shape:
            yield (f"{key}: {a.dtype} {list(a.shape)} vs "
                   f"{b.dtype} {list(b.shape)}")
        elif not torch.equal(a, b):
            yield f"{key}: {int((a != b).sum())} of {a.numel()} values differ"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_dir")
    parser.add_argument("names", nargs="*", choices=sorted(CHECKPOINTS),
                        help="checkpoints to check, all of them by default")
    parser.add_argument("--categories")
    args = parser.parse_args()
    names = args.names or sorted(CHECKPOINTS)
    failed = False
    for name in names:
        build, weights = CHECKPOINTS[name]
        file = f"{name}-{weights.name.lower().replace('_', '-')}.safetensors"
        with open(os.path.join(args.weights_dir, file), "rb") as f:
            raw = f.read()
        found = list(differences(load_safetensors(raw),
                                 build(weights=weights).state_dict()))
        verdict = "identical" if not found else f"{len(found)} differ"
        print(f"{name} {file} {len(raw)} {hashlib.sha256(raw).hexdigest()} "
              f"{verdict}")
        for line in found[:10]:
            print(f"  {line}")
        failed = failed or bool(found)
    if args.categories:
        _, weights = CHECKPOINTS[names[0]]
        with open(args.categories, "w") as out:
            out.write("\n".join(weights.meta["categories"]) + "\n")
        print(f"wrote {args.categories}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
