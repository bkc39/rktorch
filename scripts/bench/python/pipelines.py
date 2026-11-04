"""PyTorch twins of scripts/bench/workloads/pipelines.rkt.

python3 pipelines.py --cell PIPELINE WORKERS [--warmup N --epochs N --samples N]
prints one JSON record of DataLoader throughput."""

import argparse
import json
import os
import random
import resource
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


def jpeg_files():
    classes = sorted(c for c in os.listdir(HYMENOPTERA)
                     if os.path.isdir(os.path.join(HYMENOPTERA, c)))
    return [(os.path.join(HYMENOPTERA, c, f), label)
            for label, c in enumerate(classes)
            for f in sorted(os.listdir(os.path.join(HYMENOPTERA, c)))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]


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
        image = decode_jpeg(read_file(self.files[k][0]), mode=ImageReadMode.RGB)
        return self.augment(image), self.labels[k]


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


def cell(name, workers, warmup, epochs, samples):
    make, per_epoch, batch = PIPELINES[name]
    loader = DataLoader(make(samples or per_epoch), batch_size=batch, shuffle=True,
                        num_workers=workers, persistent_workers=workers > 0,
                        worker_init_fn=init_worker)
    for _ in range(warmup):
        epoch(loader)
    t0 = time.perf_counter()
    n = sum(epoch(loader) for _ in range(epochs))
    elapsed = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    return {"samples": n, "samples_per_s": n / elapsed, "elapsed_s": elapsed,
            "gc_ms": None, "peak_rss_mb": rss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", nargs=2, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    random.seed(0)
    torch.manual_seed(0)
    name, workers = args.cell
    print(json.dumps(cell(name, int(workers), args.warmup, args.epochs, args.samples)))


if __name__ == "__main__":
    main()
