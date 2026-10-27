"""PyTorch twin of scripts/bench/workloads/cifar-resnet.rkt: resnet.py's
ResNet-18, SGD and one-cycle schedule, fed by the DataLoader over the
CIFAR-10 PNG tree instead of device-resident tensors, each batch moved to
the device (non_blocking from pinned memory on CUDA)."""

from cifar_pipeline import folder_loader, folder_root
from common import StepLog, env_number, env_text, pick_device, steps_out
from resnet import ResNet, train_step

import torch


def main():
    out = steps_out()
    epochs = env_number("EPOCHS", 2)
    workers = env_number("WORKERS", 0)
    batch = env_number("BATCH", 128)
    base = env_number("BASE", 64)
    pipeline = env_text("PIPELINE", "per-sample")
    root, digest = folder_root(env_text("DATASET", "full"), env_text("PNG_DIGEST", None))
    max_lr = 0.1
    torch.backends.cudnn.benchmark = False
    device = pick_device()
    log = StepLog(out, device)
    torch.manual_seed(0)
    loader, pin = folder_loader(root, pipeline, device, batch=batch, workers=workers)
    with torch.device(device):
        net = ResNet(base=base)
    opt = torch.optim.SGD(net.parameters(), lr=max_lr, momentum=0.9,
                          nesterov=True, weight_decay=5e-4)
    log.note_optimizer("default", opt)
    log.note(pin_memory=pin, fixture_digest=digest)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lr, total_steps=epochs * len(loader),
        pct_start=0.3, div_factor=25.0, final_div_factor=1e4,
        cycle_momentum=False)
    for _ in range(epochs):
        for xb, yb in loader:
            loss = train_step(net, opt, schedule, xb.to(device, non_blocking=pin),
                              yb.to(device, non_blocking=pin))
            log.step()
            del loss
    log.finish()


if __name__ == "__main__":
    main()
