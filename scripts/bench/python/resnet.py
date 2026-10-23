"""PyTorch twin of train-cifar10 in torch/scribblings/examples/09-resnet.rkt
at its headline settings: ResNet-18 for 32x32 images, batches of 128 from
the device-resident training set, each cropped with four pixels of padding
and flipped on the device by the same per-image algorithm as
torch/vision/transforms.rkt, SGD with Nesterov momentum 0.9 and weight decay
5e-4 under a one-cycle schedule peaking at 0.1 (momentum not cycled, as
rktorch's one-cycle-lr leaves it), bfloat16 autocast on CUDA, the loss read
every step, test accuracy after each of EPOCHS epochs."""

import random

import torch.nn.functional as F
from common import (
    StepLog,
    batches,
    env_number,
    env_text,
    load_cifar10,
    optimizer_options,
    pick_device,
    steps_out,
)

import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, inp, out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.shortcut = None
        if stride != 1 or inp != out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out))

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + (self.shortcut(x) if self.shortcut else x))


def stage(inp, out, blocks, stride):
    layers = [BasicBlock(inp, out, stride)]
    layers += [BasicBlock(out, out) for _ in range(blocks - 1)]
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self, classes=10, base=64, blocks=(2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Conv2d(3, base, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(base)
        self.layer1 = stage(base, base, blocks[0], 1)
        self.layer2 = stage(base, 2 * base, blocks[1], 2)
        self.layer3 = stage(2 * base, 4 * base, blocks[2], 2)
        self.layer4 = stage(4 * base, 8 * base, blocks[3], 2)
        self.fc = nn.Linear(8 * base, classes)

    def forward(self, x):
        h = F.relu(self.bn(self.stem(x)))
        h = self.layer4(self.layer3(self.layer2(self.layer1(h))))
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.fc(h)


def batch_rng(generator):
    seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator))
    return random.Random(seed)


def random_crop(x, padding, generator):
    n, c, h, w = x.shape
    padded = x.new_zeros(n, c, h + 2 * padding, w + 2 * padding)
    padded[:, :, padding:padding + h, padding:padding + w].copy_(x)
    rng = batch_rng(generator)
    span = 2 * padding + 1
    crops = []
    for i in range(n):
        top, left = rng.randrange(span), rng.randrange(span)
        crops.append(padded[i, :, top:top + h, left:left + w])
    return torch.stack(crops)


def random_horizontal_flip(x, generator, p=0.5):
    rng = batch_rng(generator)
    flipped = [1 if rng.random() < p else 0 for _ in range(len(x))]
    mask = torch.tensor(flipped, device=x.device).ne(0).reshape(-1, 1, 1, 1)
    return torch.where(mask, x.flip(3), x)


def accuracy(net, xs, ys, chunk=500):
    net.eval()
    with torch.no_grad():
        correct = sum(
            (net(xs[i:i + chunk]).argmax(1) == ys[i:i + chunk]).sum().item()
            for i in range(0, len(xs), chunk))
    net.train()
    return correct / len(xs)


def train_step(net, opt, schedule, xs, ys, set_to_none):
    opt.zero_grad(set_to_none=set_to_none)
    if xs.device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(net(xs), ys)
    else:
        loss = F.cross_entropy(net(xs), ys)
    loss.backward()
    opt.step()
    schedule.step()
    return loss


def main():
    out = steps_out()
    epochs = env_number("EPOCHS", 30)
    batch = env_number("BATCH", 128)
    base = env_number("BASE", 64)
    max_lr = 0.1
    dataset = env_text("DATASET", "full")
    style = env_text("OPTIMIZER", "default")
    options, set_to_none = optimizer_options(style)
    torch.backends.cudnn.benchmark = False
    device = pick_device()
    log = StepLog(out, device)
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    train_x, train_y = load_cifar10("train", device, dataset)
    test_x, test_y = load_cifar10("test", device, dataset)
    with torch.device(device):
        net = ResNet(base=base)
    opt = torch.optim.SGD(net.parameters(), lr=max_lr, momentum=0.9,
                          nesterov=True, weight_decay=5e-4, **options)
    log.note_optimizer(style, opt, set_to_none)
    steps_per_epoch = (len(train_x) + batch - 1) // batch
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lr, total_steps=epochs * steps_per_epoch,
        pct_start=0.3, div_factor=25.0, final_div_factor=1e4,
        cycle_momentum=False)
    for _ in range(epochs):
        for index in batches(len(train_x), batch, g, device):
            xb = train_x.index_select(0, index)
            yb = train_y.index_select(0, index)
            augmented = random_horizontal_flip(random_crop(xb, 4, g), g)
            loss = train_step(net, opt, schedule, augmented, yb, set_to_none)
            log.step()
            loss.item()
        print(f"test acc {accuracy(net, test_x, test_y)}", flush=True)
    log.finish()


if __name__ == "__main__":
    main()
