"""Shared by the PyTorch twins of scripts/bench/workloads/cifar-loader.rkt
and cifar-resnet.rkt: the CIFAR-10 PNG tree scripts/bench/cifar-folder.rkt
writes, read by torchvision's ImageFolder through a DataLoader with
NUM_WORKERS workers and pin_memory on CUDA. PIPELINE=per-sample is
torchvision's RandomCrop(32, padding=4), RandomHorizontalFlip, ToTensor,
Normalize per image in the workers; PIPELINE=batched collates the
ToTensor'd images first and crops and flips the batch in the collate, by
the per-image algorithm of resnet.py's random_crop, then normalizes it."""

import os
import sys

from common import cache_dir
from resnet import random_crop, random_horizontal_flip
from torch.utils.data import DataLoader, default_collate
from torchvision import datasets, transforms

import torch

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def folder_root(dataset, expected_digest):
    root = os.path.join(cache_dir("RKTORCH_CIFAR10_PNG_DIR", "cifar10-png"), dataset)
    try:
        with open(os.path.join(root, "digest")) as f:
            digest = f.read().strip()
    except FileNotFoundError:
        sys.exit(f"no PNG tree at {root}: run racket scripts/bench/cifar-folder.rkt {dataset}")
    if expected_digest and not digest.startswith(expected_digest):
        sys.exit(f"the tree at {root} has digest {digest}, not {expected_digest}")
    return root, digest


def per_sample():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def augmented_batch(items):
    xs, ys = default_collate(items)
    mean = torch.tensor(CIFAR_MEAN).reshape(3, 1, 1)
    std = torch.tensor(CIFAR_STD).reshape(3, 1, 1)
    flipped = random_horizontal_flip(random_crop(xs, 4, None), None)
    return (flipped - mean) / std, ys


def folder_loader(root, pipeline, device, *, batch, workers, drop_last=False):
    train = os.path.join(root, "train")
    if pipeline == "per-sample":
        ds, collate = datasets.ImageFolder(train, transform=per_sample()), None
    elif pipeline == "batched":
        ds = datasets.ImageFolder(train, transform=transforms.ToTensor())
        collate = augmented_batch
    else:
        sys.exit(f"PIPELINE is per-sample or batched, got {pipeline!r}")
    pin = device.type == "cuda"
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                        collate_fn=collate, pin_memory=pin, drop_last=drop_last,
                        generator=torch.Generator().manual_seed(0))
    return loader, pin
