"""PyTorch twin of scripts/bench/threads.rkt: N Python threads run the same
op mix (torch releases the GIL inside each op) while the main thread sleeps
or trains the same MLP; throughput is the workers' ops per second."""

import argparse
import itertools
import threading
import time

import torch.nn.functional as F
from common import GcClock, emit, meta

import torch
from torch import nn

OPS_PER_ITERATION = 4


def op_mix():
    a8, b8 = torch.randn(8, 8), torch.randn(8, 8)
    a64, b64 = torch.randn(64, 64), torch.randn(64, 64)

    def run():
        a8 + b8
        a64 @ b64
        torch.relu(a64)
        b64.sum()

    return run


def training_step():
    net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    xs = torch.randn(64, 784)
    ys = torch.tensor([i % 10 for i in range(64)])

    def step():
        opt.zero_grad(set_to_none=False)
        F.cross_entropy(net(xs), ys).backward()
        opt.step()

    return step


def run_main(state, deadline):
    if state == "idle":
        time.sleep(max(0.0, deadline - time.perf_counter()))
        return 0
    step = training_step()
    steps = 0
    while time.perf_counter() < deadline:
        step()
        steps += 1
    return steps


def scaling_run(n, state, seconds, make_mix=op_mix):
    go = threading.Event()
    stop = threading.Event()
    counts = [0] * n
    failures = []

    def work(i, mix):
        go.wait()
        k = 0
        try:
            while k == 0 or not stop.is_set():
                mix()
                k += 1
                counts[i] = k
        except Exception as e:  # noqa: BLE001
            failures.append(e)
            stop.set()

    workers = [threading.Thread(target=work, args=(i, make_mix()), daemon=True)
               for i in range(n)]
    for w in workers:
        w.start()
    t0 = time.perf_counter()
    go.set()
    main_steps = run_main(state, t0 + seconds)
    elapsed = time.perf_counter() - t0
    ops = sum(counts)
    stop.set()
    for w in workers:
        w.join(30)
    if failures:
        raise failures[0]
    return {"ops_per_s": OPS_PER_ITERATION * ops / elapsed,
            "main_steps_per_s": main_steps / elapsed,
            "started_s": t0,
            "stopped": sum(not w.is_alive() for w in workers)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument("--states", default="idle,train")
    parser.add_argument("--intra-op", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.intra_op)
    cases = list(itertools.product(args.states.split(","),
                                   [int(n) for n in args.threads.split(",")]))
    clock = GcClock()
    for _ in range(args.warmup):
        for state, n in cases:
            scaling_run(n, state, args.seconds)
    rows = {case: [] for case in cases}
    for i in range(args.rounds):
        k = i % len(cases)
        for case in cases[k:] + cases[:k]:
            gc0 = clock.ms()
            run = scaling_run(case[1], case[0], args.seconds)
            rows[case].append((run, clock.ms() - gc0))
    device = torch.device("cpu")
    for (state, n), runs in rows.items():
        emit({"group": "threads", "case": state, "variant": str(n),
              "unit": "ops/s", "reps": 1, "threads_run": n,
              "samples": [r["ops_per_s"] for r, _ in runs],
              "main_steps_per_s": [r["main_steps_per_s"] for r, _ in runs],
              "stopped": [r["stopped"] for r, _ in runs],
              "gc_ms": [ms for _, ms in runs], "meta": meta(device)})


if __name__ == "__main__":
    main()
