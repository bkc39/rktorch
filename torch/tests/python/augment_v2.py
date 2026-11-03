"""torchvision.transforms.v2.functional colour and crop ops, for
augment-parity-test.rkt.

A seeded [2 3 12 14] float batch goes through each op at each real
parameter; the batch travels back with the outputs so the Racket side runs on
exactly it, on RKTORCH_PARITY_DEVICE. resized_crop takes integer boxes
(top, left, height, width) without antialias; for each it reports the output
rows and columns whose source coordinate lies inside the box, the pixels
where grid sampling and interpolation read the same inputs (all of them on a
downscale).
"""
import json
import os

import torch
import torchvision.transforms.v2.functional as F

from pretrained_common import packed

device = os.environ.get("RKTORCH_PARITY_DEVICE", "cpu")

FACTORS = {
    "brightness": (0.0, 0.5, 1.3),
    "contrast": (0.0, 0.6, 1.7),
    "saturation": (0.0, 0.4, 2.0),
    "hue": (-0.4, -0.1, 0.25),
    "solarize": (0.3, 0.7),
}
OPS = {
    "brightness": F.adjust_brightness,
    "contrast": F.adjust_contrast,
    "saturation": F.adjust_saturation,
    "hue": F.adjust_hue,
    "solarize": F.solarize,
}
BOXES = (((1, 2, 9, 10), (6, 5)), ((0, 0, 12, 14), (7, 7)),
         ((2, 3, 4, 5), (8, 10)), ((5, 1, 6, 3), (9, 9)))


def inside(extent, out):
    return [j for j in range(out)
            if 0 <= (j + 0.5) * extent / out - 0.5 <= extent - 1]


torch.manual_seed(0)
x = torch.rand(2, 3, 12, 14).to(device)
cases = {"x": packed(x.cpu()), "factors": FACTORS}
for name, op in OPS.items():
    cases[name] = [packed(op(x, f).cpu()) for f in FACTORS[name]]
cases["gray"] = [packed(F.rgb_to_grayscale(x, num_output_channels=k).cpu())
                 for k in (1, 3)]
cases["crops"] = []
for (top, left, height, width), size in BOXES:
    out = F.resized_crop(x, top, left, height, width, list(size),
                         antialias=False)
    cases["crops"].append({"box": [top, left, height, width],
                           "size": list(size),
                           "rows": inside(height, size[0]),
                           "cols": inside(width, size[1]),
                           "out": packed(out.cpu())})
print(json.dumps(cases))
