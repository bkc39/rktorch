"""Parity twin of examples/racket/09-resnet.rkt: the same narrow ResNet
(base width 16, blocks 2-2-2-2, bias-free convolutions, batch norm) in the
same declaration order, seeded, and 3 full-batch steps of SGD at lr 0.005 with
momentum 0.9 and weight decay 5e-4 (a gentle rate: batch norm amplifies the
last-bit kernel differences between torch builds) on the committed CIFAR-10 fixture; prints per-step
losses and the flattened parameters."""

import json
import os

import torch
import torch.nn.functional as F
from torch import nn

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "..", "torch",
                       "vision", "fixtures", "cifar10-256.bin")
DEVICE = os.environ.get("RKTORCH_PARITY_DEVICE") or "cpu"
RECORD = 3073


def load_fixture():
    with open(FIXTURE, "rb") as f:
        data = f.read()
    n = len(data) // RECORD
    raw = torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(n, RECORD)
    labels = raw[:, 0].long()
    images = raw[:, 1:].float().div(127.5).sub(1.0).reshape(n, 3, 32, 32)
    return images, labels


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


torch.manual_seed(0)
xs, ys = load_fixture()
xs, ys = xs.to(DEVICE), ys.to(DEVICE)
with torch.device(DEVICE):
    net = ResNet(base=16)
opt = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.9,
                      weight_decay=5e-4)

losses = []
for _ in range(3):
    opt.zero_grad()
    loss = F.cross_entropy(net(xs), ys)
    loss.backward()
    opt.step()
    losses.append(loss.item())

params = torch.cat([p.detach().flatten() for p in net.parameters()])
print(json.dumps({
    "shape": list(params.shape),
    "values": params.tolist(),
    "losses": losses,
}))
