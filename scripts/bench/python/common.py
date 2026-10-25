"""Shared by the PyTorch twins of scripts/bench: the e2e steps file that
training.rkt's recorder writes, the micro records harness.rkt writes, and
the meta the Racket parent completes."""

import gc
import gzip
import json
import os
import platform
import sys
import tarfile
import time

import torch


def env_number(name, default):
    supplied = os.environ.get(name)
    if supplied is None:
        return default
    try:
        value = float(supplied)
    except ValueError:
        sys.exit(f"env_number: {name} is not a real number: {supplied!r}")
    return int(value) if value.is_integer() else value


def env_text(name, default):
    return os.environ.get(name) or default


def mps_usable():
    try:
        torch.zeros(1, device="mps").to("cpu")
    except RuntimeError as e:
        sys.stderr.write(f"pick_device: MPS cannot allocate, so the CPU stands in: {e}\n")
        return False
    return True


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and mps_usable():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def meta(device):
    return {
        "impl": "pytorch",
        "python": platform.python_version(),
        "libtorch": torch.__version__,
        "device": device.type,
        "gpu": (torch.cuda.get_device_name(device)
                if device.type == "cuda" else None),
        "threads": torch.get_num_threads(),
        "threads_source": "torch.get_num_threads",
    }


class GcClock:
    """Milliseconds spent in Python's cyclic collector since start."""

    def __init__(self):
        self.total = 0.0
        self.started = None
        gc.callbacks.append(self.note)

    def note(self, phase, _info):
        now = time.perf_counter()
        if phase == "start":
            self.started = now
        elif self.started is not None:
            self.total += (now - self.started) * 1000.0
            self.started = None

    def ms(self):
        return self.total


def clock_ms():
    return time.perf_counter() * 1000.0


OPTIMIZER_STYLES = {
    "default": {"foreach": None},
    "per-param": {"foreach": False},
}


def optimizer_options(style):
    """The optimizer keywords for OPTIMIZER: "default" is PyTorch's
    (foreach kernels where it picks them), "per-param" steps each parameter
    alone, as rktorch's optimizers do."""
    if style not in OPTIMIZER_STYLES:
        sys.exit(f"OPTIMIZER is one of {sorted(OPTIMIZER_STYLES)}: {style!r}")
    return OPTIMIZER_STYLES[style]


def resolved_foreach(opt):
    requested = opt.defaults.get("foreach")
    if requested is not None:
        return requested
    params = [p for group in opt.param_groups for p in group["params"]]
    try:
        from torch.optim.optimizer import _default_to_fused_or_foreach
    except ImportError:
        return None
    _, foreach = _default_to_fused_or_foreach(params, differentiable=False,
                                              use_fused=False)
    return foreach


class StepLog:
    """One [optimizer, ms, gc_ms] row per optimizer step, the device settled
    first, as the rktorch-step event logs them."""

    def __init__(self, out, device):
        self.out = out
        self.device = device
        self.gc = GcClock()
        self.steps = []
        self.extra = {}
        self.t0 = clock_ms()
        self.write()

    def note(self, **fields):
        self.extra.update(fields)
        self.write()

    def note_optimizer(self, style, opt):
        self.note(optimizer=style, foreach=resolved_foreach(opt))

    def step(self):
        synchronize(self.device)
        self.steps.append([1, clock_ms(), self.gc.ms()])

    def write(self, wall=None):
        data = {"meta": {**meta(self.device), **self.extra},
                "steps": self.steps}
        if wall is not None:
            data["wall_ms"] = wall
        tmp = self.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self.out)

    def finish(self):
        self.write(clock_ms() - self.t0)


def steps_out():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} STEPS-FILE")
    return sys.argv[1]


def cache_dir(variable, name):
    override = os.environ.get(variable)
    if override:
        return override
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache")
    return os.path.join(base, "rktorch", name)


def idx_payload(data):
    assert data[0] == 0 and data[1] == 0 and data[2] == 8, "not a uint8 IDX"
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big")
            for i in range(ndim)]
    return dims, data[4 + 4 * ndim:]


FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "..", "torch", "vision", "fixtures")


def mnist_bytes(split, dataset):
    out = []
    for kind in ("images-idx3-ubyte", "labels-idx1-ubyte"):
        if dataset == "fixture":
            with open(os.path.join(FIXTURES, f"mnist-256-{kind}"), "rb") as f:
                out.append(idx_payload(f.read()))
        else:
            root = cache_dir("RKTORCH_MNIST_DIR", "mnist")
            prefix = "t10k" if split == "test" else "train"
            with gzip.open(os.path.join(root, f"{prefix}-{kind}.gz")) as f:
                out.append(idx_payload(f.read()))
    return out


def load_mnist(split, device, dataset):
    """MNIST from the cache rktorch's load-mnist fills, or the committed
    256-image fixture when dataset is "fixture"."""
    (dims, images), (_, labels) = mnist_bytes(split, dataset)
    n, h, w = dims
    xs = (torch.frombuffer(bytearray(images), dtype=torch.uint8)
          .float().div(255.0).reshape(n, 1, h, w))
    ys = torch.frombuffer(bytearray(labels), dtype=torch.uint8).long()
    return xs.to(device), ys.to(device)


def cifar10_bytes(split, dataset):
    if dataset == "fixture":
        with open(os.path.join(FIXTURES, "cifar10-256.bin"), "rb") as f:
            return f.read()
    root = cache_dir("RKTORCH_CIFAR10_DIR", "cifar10")
    names = (["test_batch.bin"] if split == "test"
             else [f"data_batch_{i}.bin" for i in range(1, 6)])
    with tarfile.open(os.path.join(root, "cifar-10-binary.tar.gz")) as tar:
        members = {os.path.basename(m.name): m for m in tar.getmembers()}
        return b"".join(tar.extractfile(members[n]).read() for n in names)


def load_cifar10(split, device, dataset):
    data = cifar10_bytes(split, dataset)
    raw = torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(-1, 3073)
    xs = raw[:, 1:].float().div(127.5).sub(1.0).reshape(-1, 3, 32, 32)
    return xs.to(device), raw[:, 0].long().to(device)


def batches(n, batch, generator, device):
    """A shuffled traversal as rktorch's dataloader draws it over a
    device-resident tensor-dataset: one CPU randperm, then each batch's
    indices moved to the device for one index_select per tensor."""
    order = torch.randperm(n, generator=generator)
    for start in range(0, n, batch):
        yield order[start:start + batch].to(device)


def emit(record):
    sys.stdout.write(json.dumps(record) + "\n")
    sys.stdout.flush()
