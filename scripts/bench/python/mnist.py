"""PyTorch twin of train-mnist in torch/scribblings/examples/05-mnist.rkt at
its headline settings: the convnet, Adam at 1e-3, shuffled batches of 128
drawn from the device-resident training set, test accuracy after each of
EPOCHS epochs. Every optimizer step is logged, the device settled first,
for the e2e suite to time."""

import torch.nn.functional as F
from common import (
    StepLog,
    batches,
    env_number,
    env_text,
    load_mnist,
    optimizer_options,
    pick_device,
    steps_out,
)

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(torch.relu(self.conv(x)))


def convnet():
    return nn.Sequential(ConvBlock(1, 16), ConvBlock(16, 32), nn.Flatten(),
                         nn.Linear(800, 128), nn.ReLU(), nn.Linear(128, 10))


def accuracy(net, xs, ys, chunk=1000):
    net.eval()
    with torch.no_grad():
        correct = sum(
            (net(xs[i:i + chunk]).argmax(1) == ys[i:i + chunk]).sum().item()
            for i in range(0, len(xs), chunk))
    net.train()
    return correct / len(xs)


def main():
    out = steps_out()
    epochs = env_number("EPOCHS", 3)
    batch = env_number("BATCH", 128)
    dataset = env_text("DATASET", "full")
    style = env_text("OPTIMIZER", "default")
    options, set_to_none = optimizer_options(style)
    torch.backends.cudnn.benchmark = False
    device = pick_device()
    log = StepLog(out, device)
    torch.manual_seed(0)
    train_x, train_y = load_mnist("train", device, dataset)
    test_x, test_y = load_mnist("test", device, dataset)
    with torch.device(device):
        net = convnet()
    opt = torch.optim.Adam(net.parameters(), lr=0.001, **options)
    log.note_optimizer(style, opt, set_to_none)
    generator = torch.Generator().manual_seed(0)
    for _ in range(epochs):
        for index in batches(len(train_x), batch, generator, device):
            xb = train_x.index_select(0, index)
            yb = train_y.index_select(0, index)
            opt.zero_grad(set_to_none=set_to_none)
            loss = F.cross_entropy(net(xb), yb)
            loss.backward()
            opt.step()
            log.step()
        print(f"test acc {accuracy(net, test_x, test_y)}", flush=True)
    log.finish()


if __name__ == "__main__":
    main()
