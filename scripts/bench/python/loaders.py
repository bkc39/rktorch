"""PyTorch arms of scripts/bench/m0.rkt's blocks.

python3 loaders.py loader PIPELINE WORKERS [--persistent] [--warmup N --epochs N
                                            --samples N --dataset full|fixture]
python3 loaders.py train WORKERS [--seconds S --warmup N]
print one JSON record: samples/s or main's steps/s, GC in the main process, and
the OS threads of main and of each DataLoader worker."""

import argparse
import gc
import json
import os
import platform
import resource
import sys
import threading

from cifar_pipeline import augmented_batch, folder_root, per_sample
from common import GcClock, clock_ms
from pipelines import PIPELINES, init_worker, peak_rss_mb
from threads import PeakSampler, os_thread_count, scaling_run, settled_thread_count
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import torch

PNG_BATCH = 128
SAMPLE_INTERVAL = 0.1


def spread_subset(ds, samples):
    """Every (len // samples)-th item, as cifar-pipeline.rkt's spread-subset."""
    n = len(ds)
    if not samples or samples >= n:
        return ds
    stride = n // samples
    return Subset(ds, range(0, stride * samples, stride))


def png_dataset(name, dataset, samples):
    root, _digest = folder_root(dataset, None)
    train = os.path.join(root, "train")
    if name == "cifar-png":
        return spread_subset(datasets.ImageFolder(train, transform=per_sample()), samples), None
    images = datasets.ImageFolder(train, transform=transforms.ToTensor())
    return spread_subset(images, samples), augmented_batch


def m0_loader(name, workers, persistent, samples, dataset):
    if name in ("cifar-png", "cifar-batched"):
        ds, collate = png_dataset(name, dataset, samples)
        batch = PNG_BATCH
    else:
        make, per_epoch, batch = PIPELINES[name]
        ds, collate = make(samples or per_epoch), None
    return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                      persistent_workers=persistent and workers > 0,
                      worker_init_fn=init_worker, collate_fn=collate,
                      generator=torch.Generator().manual_seed(0))


def task_dir(pid):
    return f"/proc/{pid}/task"


class WorkerThreads:
    """The most OS threads each worker process seen so far ran at once."""

    def __init__(self, interval=SAMPLE_INTERVAL, tasks=task_dir):
        self.tasks = tasks
        self.interval = interval
        self.peaks = {}
        self.pids = []
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def watch(self, pids):
        with self.lock:
            self.pids = list(pids)
        self.sample()

    def sample(self):
        with self.lock:
            pids = list(self.pids)
        for pid in pids:
            count = os_thread_count(self.tasks(pid))
            if count is not None:
                self.peaks[pid] = max(count, self.peaks.get(pid, 0))

    def run(self):
        while not self.done.wait(self.interval):
            self.sample()

    def stop(self):
        self.done.set()
        self.thread.join()
        self.sample()
        return list(self.peaks.values())


def epoch(loader, workers):
    it = iter(loader)
    workers.watch(w.pid for w in getattr(it, "_workers", ()))
    n = 0
    for batch in it:
        batch[0].sum()
        n += batch[0].shape[0]
    return n


def gc_count():
    return sum(s["collections"] for s in gc.get_stats())


def measured(run):
    """run() under the clock, the main process's collector and an OS-thread
    sampler; run receives the worker sampler."""
    baseline = settled_thread_count()
    clock = GcClock()
    count0 = gc_count()
    main = PeakSampler()
    workers = WorkerThreads()
    t0 = clock_ms()
    try:
        result = run(workers)
    finally:
        elapsed = (clock_ms() - t0) / 1000.0
        peak = main.stop()
        worker_peaks = workers.stop()
    return result, {"elapsed_s": elapsed, "gc_ms": clock.ms(), "gc_count": gc_count() - count0,
                    "os_threads_baseline": baseline, "os_threads_peak": peak,
                    "worker_threads": worker_peaks}


def process_fields(stats):
    rss = peak_rss_mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, sys.platform)
    return {**stats, "python": platform.python_version(), "libtorch": torch.__version__,
            "intra_op_threads": torch.get_num_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "peak_rss_mb": rss}


def loader_cell(name, workers, persistent, warmup, epochs, samples, dataset):
    loader = m0_loader(name, workers, persistent, samples, dataset)
    idle = WorkerThreads()
    for _ in range(warmup):
        epoch(loader, idle)
    idle.stop()
    n, stats = measured(lambda sampler: sum(epoch(loader, sampler) for _ in range(epochs)))
    return {**process_fields(stats), "samples": n, "samples_per_s": n / stats["elapsed_s"],
            "persistent": persistent and workers > 0, "warmup": warmup, "epochs": epochs}


def train_cell(workers, seconds, warmup):
    for _ in range(warmup):
        scaling_run(workers, "train", seconds)
    run, stats = measured(lambda _sampler: scaling_run(workers, "train", seconds))
    return {**process_fields(stats), "main_steps_per_s": run["main_steps_per_s"],
            "ops_per_s": run["ops_per_s"], "seconds": seconds, "warmup": warmup}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["loader", "train"])
    parser.add_argument("cell", nargs="+")
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--dataset", default="full")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    if args.kind == "loader":
        name, workers = args.cell
        record = loader_cell(name, int(workers), args.persistent, args.warmup, args.epochs,
                             args.samples, args.dataset)
    else:
        (workers,) = args.cell
        record = train_cell(int(workers), args.seconds, args.warmup)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
