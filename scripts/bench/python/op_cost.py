"""PyTorch twin of scripts/bench/op-cost.rkt: the same ops on the main thread,
then on N Python threads running the same loop; one JSON record per
condition, in the scoreboard's schema."""

import argparse
import os
import threading
import time

from common import GcClock, emit
from pipelines import cifar_augment, cifar_sources

import torch


def scalar_add():
    x = torch.zeros(1)
    return lambda: x + 1.0


def add_8x8():
    a, b = torch.randn(8, 8), torch.randn(8, 8)
    return lambda: a + b


def zeros_8x8():
    return lambda: torch.zeros(8, 8)


def narrow():
    x = torch.randn(3, 40, 40)
    return lambda: x.narrow(1, 4, 32)


def stack_inputs():
    return [torch.randn(3, 32, 32) for _ in range(32)]


def stack_32():
    ts = stack_inputs()
    return lambda: torch.stack(ts)


def cifar_chain():
    images, _ = cifar_sources()
    augment = cifar_augment()
    state = {"k": 0}

    def run():
        state["k"] = (state["k"] + 1) % len(images)
        return augment(images[state["k"]].to(torch.float32) / 255.0)
    return run


OPS = {"scalar-add": scalar_add, "add-8x8": add_8x8, "zeros-8x8": zeros_8x8,
       "narrow": narrow, "stack-32": stack_32, "cifar-chain": cifar_chain}


def main_loop(step, seconds):
    t0 = time.perf_counter()
    deadline = t0 + seconds
    n = 0
    while time.perf_counter() < deadline:
        step()
        n += 1
    return n, time.perf_counter() - t0


def thread_loop(n, step, seconds):
    counts = [0] * n
    go = threading.Event()
    stop = threading.Event()

    errors = []

    def work(i):
        go.wait()
        k = 0
        try:
            while not stop.is_set():
                step()
                k += 1
                counts[i] = k
        except BaseException as e:
            errors.append(e)
            stop.set()

    workers = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    for w in workers:
        w.start()
    t0 = time.perf_counter()
    go.set()
    stop.wait(seconds)
    stop.set()
    elapsed = time.perf_counter() - t0
    ops = sum(counts)
    for w in workers:
        w.join()
    if errors:
        raise errors[0]
    return ops, elapsed


def load():
    return list(os.getloadavg())


def sample(op, threads, step, seconds, gc_clock):
    gc0 = gc_clock.ms()
    run_load = load()
    ops, elapsed = main_loop(step, seconds) if threads == 0 else thread_loop(threads, step, seconds)
    return {"kind": "op", "impl": "pytorch", "op": op, "layer": "facade", "threads": threads,
            "ops": ops, "elapsed_ms": 1000.0 * elapsed,
            "us_per_op": 1e6 * elapsed * max(1, threads) / ops if ops else None,
            "ops_per_s": ops / elapsed, "heap_bytes_per_op": None,
            "gc_ms": gc_clock.ms() - gc0, "load": run_load}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=0.5)
    parser.add_argument("--thread-seconds", type=float, default=1.0)
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument("--ops", default=",".join(OPS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    gc_clock = GcClock()
    steps = {op: OPS[op]() for op in args.ops.split(",")}
    for step in steps.values():
        main_loop(step, 0.1)
    threads = [0] + [int(n) for n in args.threads.split(",")]
    conditions = [(op, n) for n in threads for op in steps]
    k = args.rep % len(conditions)
    for op, n in conditions[k:] + conditions[:k]:
        record = sample(op, n, steps[op], args.seconds if n == 0 else args.thread_seconds, gc_clock)
        record["rep"] = args.rep
        emit(record)


if __name__ == "__main__":
    main()
