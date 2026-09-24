"""Exports torchvision's pretrained ImageNet weights as safetensors, the
files torch/vision/weights.rkt fetches from this repository's
`weights-v1` release:

    nix develop --command python3 scripts/export-weights.py OUT_DIR [NAME ...]

Each file is the model's state_dict verbatim, keys in state_dict order,
so its names are torchvision's (`bn1.running_mean`, `layer1.0.downsample.0
.weight`) and weights.rkt renames them on load. The header carries the
source URL and the licence. The bytes depend only on the weights and on
the versions recorded in the header, so a rerun reproduces the checksums
in weights.rkt; the script prints them.

`--categories PATH` also writes the ImageNet class names, one per line,
in label order, from the same weights' metadata.
"""
import argparse
import hashlib
import json
import os
import struct

import torch
import torchvision
from torchvision import models

CHECKPOINTS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
}

TAGS = {torch.float32: "F32", torch.float64: "F64", torch.float16: "F16",
        torch.bfloat16: "BF16", torch.int64: "I64", torch.uint8: "U8",
        torch.bool: "BOOL"}

LICENSE = ("BSD-3-Clause (torchvision, Copyright (c) Soumith Chintala 2016); "
           "trained on ImageNet-1K, whose terms of access bind the user")


def safetensors_bytes(state, metadata):
    header = {"__metadata__": metadata}
    chunks = []
    offset = 0
    for key, value in state.items():
        t = value.detach().cpu().contiguous()
        data = t.numpy().tobytes()
        header[key] = {"dtype": TAGS[t.dtype], "shape": list(t.shape),
                       "data_offsets": [offset, offset + len(data)]}
        chunks.append(data)
        offset += len(data)
    head = json.dumps(header, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)
    return struct.pack("<Q", len(head)) + head + b"".join(chunks)


def file_name(name, weights):
    return f"{name}-{weights.name.lower().replace('_', '-')}.safetensors"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("names", nargs="*", default=sorted(CHECKPOINTS))
    parser.add_argument("--categories")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for name in args.names:
        build, weights = CHECKPOINTS[name]
        model = build(weights=weights).eval()
        metadata = {
            "source": weights.url,
            "weights": f"torchvision.models.{type(weights).__name__}."
                       f"{weights.name}",
            "torchvision": torchvision.__version__.split("+")[0],
            "license": LICENSE,
        }
        data = safetensors_bytes(model.state_dict(), metadata)
        path = os.path.join(args.out_dir, file_name(name, weights))
        with open(path, "wb") as out:
            out.write(data)
        digest = hashlib.sha256(data).hexdigest()
        print(f"{name} {os.path.basename(path)} {len(data)} {digest}")
    if args.categories:
        _, weights = CHECKPOINTS[args.names[0]]
        with open(args.categories, "w") as out:
            out.write("\n".join(weights.meta["categories"]) + "\n")
        print(f"wrote {args.categories}")


if __name__ == "__main__":
    main()
