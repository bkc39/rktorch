"""Per-step time of the jpeg chain at w0, the torchvision twin of
scripts/bench/jpeg-steps.rkt.

python3 jpeg_steps.py [PASSES] prints one JSON record: microseconds per
sample for decode, each transform of the chain, and resize to 256 of the
whole decoded image as uint8 and as float32."""

import json
import sys
import time

import torch
from pipelines import IMAGENET_MEAN, IMAGENET_STD, decode_rgb, jpeg_files
from torchvision.transforms import v2

CHAIN = [("random-resized-crop", v2.RandomResizedCrop(224, antialias=True)),
         ("flip", v2.RandomHorizontalFlip()),
         ("convert", v2.ToDtype(torch.float32, scale=True)),
         ("normalize", v2.Normalize(IMAGENET_MEAN, IMAGENET_STD))]
RESIZE = v2.Resize(256, antialias=True)
TO_FLOAT = v2.ToDtype(torch.float32, scale=True)
ASIDES = [("resize-256-uint8", lambda decoded, _float: RESIZE(decoded)),
          ("resize-256-float32", lambda _decoded, flt: RESIZE(flt))]
STEPS = ["decode"] + [name for name, _ in CHAIN] + [name for name, _ in ASIDES]


def timed(totals, name, thunk):
    t0 = time.perf_counter()
    y = thunk()
    if totals is not None:
        totals[name] += time.perf_counter() - t0
    return y


def run_pass(paths, totals):
    for path in paths:
        decoded = timed(totals, "decode", lambda: decode_rgb(path))
        x = decoded
        for name, f in CHAIN:
            x = timed(totals, name, lambda: f(x))
        flt = TO_FLOAT(decoded)
        for name, f in ASIDES:
            timed(totals, name, lambda: f(decoded, flt))


def main():
    passes = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    torch.set_num_threads(1)
    torch.manual_seed(0)
    paths = [path for path, _ in jpeg_files()]
    run_pass(paths, None)
    totals = {name: 0.0 for name in STEPS}
    t0 = time.perf_counter()
    for _ in range(passes):
        run_pass(paths, totals)
    wall = time.perf_counter() - t0
    n = passes * len(paths)
    print(json.dumps({"impl": "pytorch", "samples": n,
                      "wall_us_per_sample": 1e6 * wall / n,
                      "steps": {k: 1e6 * v / n for k, v in totals.items()}}))


if __name__ == "__main__":
    main()
