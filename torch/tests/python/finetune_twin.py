"""The two phases of examples/racket/15-finetune.rkt at fixture scale, for
imagenet-parity-test.rkt: torchvision's ResNet-18 with its ImageNet
weights and a fresh two-way head, three SGD steps on the head alone with
the backbone frozen, then three on everything at a tenth of the rate.
The batch is the four committed photographs through the ImageNet
preprocessing, with no augmentation, so every step is deterministic. The
fresh head travels back so the Racket side starts from the same network.
"""
import json

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torchvision import models

from pretrained_common import packed, photos, preprocess, weights

torch.manual_seed(0)
net = models.resnet18()
net.load_state_dict(weights("resnet18"))
net.fc = nn.Linear(512, 2)
head = {"weight": packed(net.fc.weight), "bias": packed(net.fc.bias)}

names, pixels = photos()
batch = torch.stack([preprocess(p) for p in pixels])
labels = torch.tensor([0 if n.startswith("ants") else 1 for n in names])
net.train()

losses = []


def run(params, lr, steps):
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    for _ in range(steps):
        opt.zero_grad()
        loss = nnf.cross_entropy(net(batch), labels)
        loss.backward()
        opt.step()
        losses.append(loss.item())


for name, p in net.named_parameters():
    p.requires_grad_(name.startswith("fc."))
run([net.fc.weight, net.fc.bias], 0.001, 3)
for p in net.parameters():
    p.requires_grad_(True)
run(list(net.parameters()), 0.0001, 3)

state = net.state_dict()
print(json.dumps({
    "head": head,
    "losses": losses,
    "after": {key: packed(state[key]) for key in
              ["fc.weight", "fc.bias", "conv1.weight",
               "layer4.1.conv2.weight", "bn1.running_mean",
               "layer4.1.bn2.running_var"]},
}))
