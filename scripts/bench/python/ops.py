"""PyTorch twin of scripts/bench/ops.rkt: the same ops at the same shapes and
repetitions, eager, timed per round as harness.rkt times a case, with the
device synchronized at the end of each round."""

import argparse
import gc
import time

import torch
import torch.nn.functional as F
from torch import nn

from common import GcClock, emit, meta, synchronize


def op_cases(device, scale):
    def n(reps):
        return max(1, round(reps * scale))

    def pair(*dims):
        return (torch.randn(*dims, device=device),
                torch.randn(*dims, device=device))

    a8, b8 = pair(8, 8)
    a64, b64 = pair(64, 64)
    a512, b512 = pair(512, 512)
    a2048, b2048 = pair(2048, 2048)
    x1 = torch.randn(1, 3, 32, 32, device=device)
    x128 = torch.randn(128, 3, 32, 32, device=device)
    w = torch.randn(16, 3, 3, 3, device=device)
    b = torch.randn(16, device=device)
    layer = nn.Linear(64, 64).to(device)
    xl = torch.randn(32, 64, device=device)

    def linear_step():
        layer(xl).sum().backward()
        return layer.weight

    return [
        ("add-8x8", n(20000), lambda: a8 + b8),
        ("matmul-64", n(5000), lambda: a64 @ b64),
        ("matmul-512", n(100), lambda: a512 @ b512),
        ("matmul-2048", n(5), lambda: a2048 @ b2048),
        ("conv2d-1x3x32x32-k16", n(2000),
         lambda: F.conv2d(x1, w, b, padding=1)),
        ("conv2d-128x3x32x32-k16", n(100),
         lambda: F.conv2d(x128, w, b, padding=1)),
        ("linear-64-fwd-bwd", n(2000), linear_step),
    ]


def run_once(call, reps, device, clock):
    gc0 = clock.ms()
    t0 = time.perf_counter()
    for _ in range(reps):
        call()
    synchronize(device)
    elapsed = time.perf_counter() - t0
    return elapsed * 1e9 / reps, clock.ms() - gc0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    clock = GcClock()
    for name, reps, call in op_cases(device, args.scale):
        gc.collect()
        for _ in range(args.warmup):
            run_once(call, reps, device, clock)
        rows = [run_once(call, reps, device, clock)
                for _ in range(args.rounds)]
        emit({"group": "ops", "case": name, "variant": "pytorch",
              "unit": "ns", "reps": reps,
              "samples": [ns for ns, _ in rows],
              "gc_ms": [ms for _, ms in rows], "meta": meta(device)})


if __name__ == "__main__":
    main()
