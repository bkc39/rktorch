"""Helpers the pretrained parity twins share: the committed photographs as
torchvision decodes them, the ImageNet preprocessing, a safetensors
reader that needs no package, and hex payloads for the JSON reply."""
import json
import os
import struct

import torch
import torchvision.transforms.functional as F
from torchvision.io import ImageReadMode, decode_image, read_file

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "vision", "fixtures", "hymenoptera")
DTYPES = {"F32": torch.float32, "I64": torch.int64}


def photos():
    paths = sorted(os.path.join(root, name)
                   for root, _, names in os.walk(FIXTURES)
                   for name in names if name.endswith(".jpg"))
    return ([os.path.relpath(p, FIXTURES) for p in paths],
            [decode_image(read_file(p), ImageReadMode.RGB) for p in paths])


def preprocess(pixels):
    return F.normalize(
        F.center_crop(F.resize(F.convert_image_dtype(pixels), 256), 224),
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def load_safetensors(path):
    with open(path, "rb") as f:
        raw = f.read()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n])
    header.pop("__metadata__", None)
    body = raw[8 + n:]
    return {k: torch.frombuffer(bytearray(body[v["data_offsets"][0]:
                                               v["data_offsets"][1]]),
                                dtype=DTYPES[v["dtype"]]).reshape(v["shape"])
            for k, v in header.items()}


def weights(name):
    return load_safetensors(os.path.join(
        os.environ["RKTORCH_PARITY_WEIGHTS"],
        f"{name}-imagenet1k-v1.safetensors"))


def packed(t):
    t = t.detach().contiguous()
    return {"shape": list(t.shape), "hex": t.numpy().tobytes().hex()}
