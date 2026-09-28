"""torchvision.io's decoder and torchvision.transforms.functional's resize,
center_crop, normalize and convert_image_dtype on the committed image
fixtures, for the Racket read-image and transforms in image-parity-test.rkt.

Pixel and float payloads travel as hex of their little-endian bytes.
"""
import json
import os

import torch
import torch.nn.functional as nnf
import torchvision.transforms.functional as F
from torchvision.io import ImageReadMode, decode_image, read_file

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "vision", "fixtures", "images")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def fixture(name):
    return read_file(os.path.join(FIXTURES, name))


def packed(t):
    return {"shape": list(t.shape),
            "hex": t.contiguous().numpy().tobytes().hex()}


decoded = {}
for name in sorted(os.listdir(FIXTURES)):
    data = fixture(name)
    decoded[name] = {
        "unchanged": packed(decode_image(data, ImageReadMode.UNCHANGED)),
        "rgb": packed(decode_image(data, ImageReadMode.RGB)),
    }

gradient = decode_image(fixture("gradient.png")).float() / 255
resizes = []
for size in [10, [50, 70], [23, 5], [7, 37], [23, 37], 40]:
    for antialias in [True, False]:
        out = F.resize(gradient, size, antialias=antialias)
        plain = nnf.interpolate(gradient.unsqueeze(0), size=list(out.shape[-2:]),
                                mode="bilinear", align_corners=False,
                                antialias=antialias).squeeze(0)
        assert torch.equal(out, plain)
        resizes.append({"size": size, "antialias": antialias, **packed(out)})

crops = [{"size": size, **packed(F.center_crop(gradient, size))}
         for size in [10, [5, 20], [23, 37], [4, 4], [22, 36]]]

photo = decode_image(fixture("smooth-401x299.jpg"))
chain = F.normalize(F.center_crop(F.resize(F.convert_image_dtype(photo), 256), 224),
                    MEAN, STD)

print(json.dumps({
    "decoded": decoded,
    "resize": resizes,
    "crop": crops,
    "normalize": packed(F.normalize(gradient, MEAN, STD)),
    "to_uint8": packed(F.convert_image_dtype(gradient * 0.75, torch.uint8)),
    "chain": packed(chain),
}))
