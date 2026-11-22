"""torchvision.io's decoder and torchvision.transforms.functional's resize,
center_crop, normalize and convert_image_dtype on the committed image
fixtures, for the Racket read-image and transforms in image-parity-test.rkt;
v2's resize and resized_crop on uint8, float and half images, and the CPU
capability that picks v2's uint8 path.

Pixel and float payloads travel as hex of their little-endian bytes.
"""
import json
import os
import platform

import torch
import torch.nn.functional as nnf
import torchvision.transforms.functional as F
import torchvision.transforms.v2.functional as F2
from torchvision.io import ImageReadMode, decode_image, read_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2.functional._geometry import (
    _do_native_uint8_resize_on_cpu)

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "vision", "fixtures", "images")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def fixture(name):
    return read_file(os.path.join(FIXTURES, name))


def packed(t):
    return {"shape": list(t.shape),
            "hex": t.contiguous().flatten().view(torch.uint8).numpy()
            .tobytes().hex()}


images = sorted(name for name in os.listdir(FIXTURES)
                if name.endswith((".jpg", ".png")))
decoded = {}
for name in images:
    data = fixture(name)
    decoded[name] = {
        "unchanged": packed(decode_image(data, ImageReadMode.UNCHANGED)),
        "rgb": packed(decode_image(data, ImageReadMode.RGB)),
        "gray": packed(decode_image(data, ImageReadMode.GRAY)),
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

g8 = decode_image(fixture("gradient.png"))
photo_rgb = decode_image(fixture("smooth-401x299.jpg"), ImageReadMode.RGB)
SIZES = [10, [50, 70], [23, 5], [7, 37], [23, 37], 40, [300, 400], [1, 1]]
WINDOWS = [(0, 0, 299, 401), (10, 20, 150, 200), (100, 150, 37, 53),
           (298, 400, 1, 1), (0, 0, 299, 100)]
ANTIALIAS = [True, False]

resize_u8 = [{"src": src, "size": size, "antialias": aa,
              **packed(F2.resize(img, size, antialias=aa))}
             for src, img in (("gradient", g8), ("photo", photo_rgb))
             for size in SIZES + [256, [224, 224]] for aa in ANTIALIAS]

resized_crop_u8 = [{"window": list(w), "size": size, "antialias": aa,
                    **packed(F2.resized_crop(photo_rgb, *w, [size, size],
                                             antialias=aa))}
                   for w in WINDOWS for size in (224, 16) for aa in ANTIALIAS]

resize_u8_float_path = [{"size": size, "antialias": aa,
                         **packed(F.resize(g8, size, antialias=aa))}
                        for size in SIZES for aa in ANTIALIAS]

resize_f64 = [{"size": size, "antialias": aa,
               **packed(F2.resize(gradient.double(), size, antialias=aa))}
              for size in SIZES for aa in ANTIALIAS]

photo_f32 = F2.to_dtype(photo_rgb, torch.float32, scale=True)
resized_crop_f32 = [{"window": list(w), "size": size, "antialias": aa,
                     **packed(F2.resized_crop(photo_f32, *w, [size, size],
                                              antialias=aa))}
                    for w in WINDOWS for size in (224, 16) for aa in ANTIALIAS]

resize_half = [{"dtype": name, "size": size, "antialias": aa,
                **packed(F2.resize(gradient.to(dtype), size, antialias=aa))}
               for name, dtype in (("bfloat16", torch.bfloat16),
                                   ("float16", torch.float16))
               for size in SIZES for aa in ANTIALIAS]

print(json.dumps({
    "capability": torch.backends.cpu.get_cpu_capability().lower(),
    "machine": platform.machine(),
    "native_uint8": _do_native_uint8_resize_on_cpu(InterpolationMode.BILINEAR),
    "resize_u8": resize_u8,
    "resized_crop_u8": resized_crop_u8,
    "resize_u8_float_path": resize_u8_float_path,
    "resize_f64": resize_f64,
    "resized_crop_f32": resized_crop_f32,
    "resize_half": resize_half,
    "decoded": decoded,
    "resize": resizes,
    "crop": crops,
    "normalize": packed(F.normalize(gradient, MEAN, STD)),
    "to_uint8": packed(F.convert_image_dtype(gradient * 0.75, torch.uint8)),
    "chain": packed(chain),
}))
