"""PyTorch twin of scripts/bench/workloads/cifar-loader.rkt: EPOCHS passes
over the CIFAR-10 PNG tree in shuffled batches of BATCH, the last partial
batch dropped, each moved to the device (non_blocking from pinned memory on
CUDA) and logged as one step, the device settled first."""

from cifar_pipeline import folder_loader, folder_root
from common import StepLog, env_number, env_text, pick_device, steps_out

import torch


def main():
    out = steps_out()
    epochs = env_number("EPOCHS", 2)
    workers = env_number("WORKERS", 0)
    batch = env_number("BATCH", 128)
    pipeline = env_text("PIPELINE", "per-sample")
    root, digest = folder_root(env_text("DATASET", "full"), env_text("PNG_DIGEST", None))
    device = pick_device()
    log = StepLog(out, device)
    torch.manual_seed(0)
    loader, pin = folder_loader(root, pipeline, device, batch=batch, workers=workers,
                                drop_last=True)
    log.note(pin_memory=pin, fixture_digest=digest)
    for _ in range(epochs):
        for xb, yb in loader:
            x = xb.to(device, non_blocking=pin)
            y = yb.to(device, non_blocking=pin)
            log.step()
            del x, y
    log.finish()


if __name__ == "__main__":
    main()
