"""PyTorch twins of scripts/bench/workloads/pipelines.rkt.

python3 -P pipelines.py --cell PIPELINE WORKERS prints one JSON record of
DataLoader throughput (2 warm-up epochs, 3 timed); --ops PIPELINE prints the
per-op timings scripts/bench/pipeline-ops.rkt prints for rktorch."""

import json
import os
import random
import resource
import sys
import time

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_jpeg, read_file
from torchvision.transforms import v2

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HYMENOPTERA = os.path.expanduser(
    "~/.cache/rktorch/hymenoptera/hymenoptera_data/train")
CIFAR = os.path.join(REPO, "torch/vision/fixtures/cifar10-256.bin")
TEXT = os.path.join(REPO, "torch/text/fixtures/heart-of-darkness-excerpt.txt")
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CIFAR_MEAN, CIFAR_STD = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
WINDOW = 128
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", "2"))
TIMED_EPOCHS = int(os.environ.get("TIMED_EPOCHS", "3"))


def jpeg_files():
    classes = sorted(c for c in os.listdir(HYMENOPTERA)
                     if os.path.isdir(os.path.join(HYMENOPTERA, c)))
    return [(os.path.join(HYMENOPTERA, c, f), label)
            for label, c in enumerate(classes)
            for f in sorted(os.listdir(os.path.join(HYMENOPTERA, c)))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]


def decode(data):
    return decode_jpeg(data, mode=ImageReadMode.RGB)


def jpeg_augment():
    return v2.Compose([v2.RandomResizedCrop(224, antialias=True),
                       v2.RandomHorizontalFlip(),
                       v2.ToDtype(torch.float32, scale=True),
                       v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def cifar_augment():
    return v2.Compose([v2.RandomCrop(32, padding=4),
                       v2.RandomHorizontalFlip(),
                       v2.Normalize(CIFAR_MEAN, CIFAR_STD)])


def cifar_sources():
    raw = torch.from_numpy(np.fromfile(CIFAR, dtype=np.uint8).reshape(-1, 3073))
    return raw[:, 1:].reshape(-1, 3, 32, 32).clone(), raw[:, 0].to(torch.int64)


def text_sources():
    with open(TEXT, encoding="utf-8") as f:
        text = f.read()
    vocab = sorted(set(text))
    index = {c: i for i, c in enumerate(vocab)}
    return torch.tensor([index[c] for c in text], dtype=torch.int64), len(vocab)


def text_window(ids, start, mask_id):
    span = ids[start:start + WINDOW + 1]
    x = torch.where(torch.rand(WINDOW) < 0.15, mask_id, span[:WINDOW])
    return x, span[1:]


class JpegCycle(Dataset):
    def __init__(self, n):
        self.files = jpeg_files()
        self.labels = torch.tensor([label for _, label in self.files])
        self.augment = jpeg_augment()
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        k = i % len(self.files)
        image = decode(read_file(self.files[k][0]))
        return self.augment(image), self.labels[k]


class JpegMem(JpegCycle):
    def __init__(self, n):
        super().__init__(n)
        # bytes, not tensors: forkserver workers cannot unpickle 244 shared fds
        self.blobs = [read_file(path).numpy().tobytes() for path, _ in self.files]

    def __getitem__(self, i):
        k = i % len(self.files)
        blob = torch.frombuffer(self.blobs[k], dtype=torch.uint8)
        return self.augment(decode(blob)), self.labels[k]


class CifarCycle(Dataset):
    def __init__(self, n):
        self.images, self.labels = cifar_sources()
        self.augment = cifar_augment()
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        k = i % len(self.images)
        pixels = self.images[k].to(torch.float32) / 255.0
        return self.augment(pixels), self.labels[k]


class MaskedText(Dataset):
    def __init__(self, n):
        self.ids, self.mask_id = text_sources()
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, _i):
        start = random.randrange(len(self.ids) - WINDOW)
        return text_window(self.ids, start, self.mask_id)


PIPELINES = {"jpeg": (JpegCycle, 2048, 32),
             "jpeg-mem": (JpegMem, 2048, 32),
             "jpeg-parts": (JpegCycle, 2048, 32),
             "cifar": (CifarCycle, 8192, 128),
             "text": (MaskedText, 8192, 64)}


def init_worker(_worker_id):
    torch.set_num_threads(1)
    random.seed(torch.initial_seed() % 2 ** 32)


def epoch(loader):
    n = 0
    for batch in loader:
        batch[0].sum()
        n += batch[0].shape[0]
    return n


def cell(name, workers):
    make, per_epoch, batch = PIPELINES[name]
    loader = DataLoader(make(per_epoch), batch_size=batch, shuffle=True,
                        num_workers=workers, persistent_workers=workers > 0,
                        worker_init_fn=init_worker)
    for _ in range(WARMUP_EPOCHS):
        epoch(loader)
    t0 = time.perf_counter()
    n = sum(epoch(loader) for _ in range(TIMED_EPOCHS))
    elapsed = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    return {"samples": n, "samples/s": n / elapsed, "elapsed-s": elapsed,
            "peak-rss-mb": rss}


def timed(step, arg, min_s=1.0, min_iters=20):
    step(arg)
    iters, t0 = 0, time.perf_counter()
    while True:
        step(arg)
        iters += 1
        elapsed = time.perf_counter() - t0
        if elapsed >= min_s and iters >= min_iters:
            return 1e6 * elapsed / iters, iters


def cycling(xs, f):
    state = {"k": 0}

    def step(_):
        state["k"] = (state["k"] + 1) % len(xs)
        return f(xs[state["k"]])
    return step


def jpeg_steps():
    files = [path for path, _ in jpeg_files()][:32]
    blobs = [read_file(p) for p in files]
    images = [decode(b) for b in blobs]
    floats = [v2.ToDtype(torch.float32, scale=True)(i) for i in images]
    crop, flip = v2.RandomResizedCrop(224, antialias=True), v2.RandomHorizontalFlip()
    to_float = v2.ToDtype(torch.float32, scale=True)
    norm = v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    crop_u8 = crop(images[0])
    out = norm(to_float(crop_u8))
    augment = jpeg_augment()
    batch = [out] * 32
    return [("read", cycling(files, read_file)),
            ("decode", cycling(blobs, decode)),
            ("to-float (full image)", cycling(images, to_float)),
            ("rrc (float input)", cycling(floats, crop)),
            ("rrc (uint8 input, twin order)", cycling(images, crop)),
            ("to-float (224 crop, twin order)", lambda _: to_float(crop_u8)),
            ("flip", lambda _: flip(out)),
            ("normalize", lambda _: norm(out)),
            ("sample (read..normalize)",
             cycling(files, lambda p: augment(decode(read_file(p))))),
            ("collate (stack 32)", lambda _: torch.stack(batch))]


def cifar_steps():
    images, _ = cifar_sources()
    pixels = images[0].to(torch.float32) / 255.0
    crop, flip = v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()
    norm = v2.Normalize(CIFAR_MEAN, CIFAR_STD)
    augment = cifar_augment()
    batch = [pixels] * 128
    ks = list(range(len(images)))
    return [("select", cycling(ks, lambda k: images[k])),
            ("to-float /255", lambda _: images[0].to(torch.float32) / 255.0),
            ("random crop pad 4", lambda _: crop(pixels)),
            ("flip", lambda _: flip(pixels)),
            ("normalize", lambda _: norm(pixels)),
            ("sample (select..normalize)",
             cycling(ks, lambda k: augment(images[k].to(torch.float32) / 255.0))),
            ("collate (stack 128)", lambda _: torch.stack(batch))]


def text_steps():
    ids, mask_id = text_sources()
    span = ids[0:WINDOW + 1]
    x = span[:WINDOW]
    r = torch.rand(WINDOW)
    mask = r < 0.15
    batch = [x.clone()] * 64
    m = len(ids) - WINDOW
    return [("narrow 129", lambda _: ids[random.randrange(m):][:WINDOW + 1]),
            ("rand 128", lambda _: torch.rand(WINDOW)),
            ("< 0.15", lambda _: r < 0.15),
            ("where", lambda _: torch.where(mask, mask_id, x)),
            ("narrow x, y", lambda _: (span[:WINDOW], span[1:])),
            ("sample (window)",
             lambda _: text_window(ids, random.randrange(m), mask_id)),
            ("collate (stack 64, x and y)",
             lambda _: (torch.stack(batch), torch.stack(batch)))]


def overhead_steps():
    small = torch.zeros(1)
    return [("add scalar to [1]", lambda _: small + 1.0)]


def ops(name):
    steps = {"jpeg": jpeg_steps, "cifar": cifar_steps, "text": text_steps,
             "overhead": overhead_steps}[name]()
    return [{"pipeline": name, "op": op, "us": us, "iters": iters}
            for op, step in steps
            for us, iters in [timed(step, None)]]


def main():
    torch.set_num_threads(1)
    random.seed(0)
    torch.manual_seed(0)
    mode, name = sys.argv[1], sys.argv[2]
    if mode == "--cell":
        print(json.dumps(cell(name, int(sys.argv[3]))))
    elif mode == "--ops":
        for record in ops(name):
            print(json.dumps(record))
    else:
        sys.exit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
