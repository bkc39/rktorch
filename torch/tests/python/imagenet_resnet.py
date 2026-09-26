"""torchvision's ImageNet ResNets on the committed photographs, with the
weights read from the same safetensors files the Racket side loads
(RKTORCH_PARITY_WEIGHTS names their directory), for
imagenet-parity-test.rkt. Each photograph travels back as torchvision's
decoded pixels, so the Racket side runs the networks on exactly this
input, beside the logits and top five of every model.
"""
import json

import torch
from torchvision import models

from pretrained_common import packed, photos, preprocess, weights

MODELS = {"resnet18": models.resnet18, "resnet34": models.resnet34,
          "resnet50": models.resnet50}

names, pixels = photos()
batch = torch.stack([preprocess(p) for p in pixels])

logits = {}
for name, build in MODELS.items():
    net = build()
    net.load_state_dict(weights(name))
    with torch.no_grad():
        logits[name] = net.eval()(batch)

print(json.dumps({
    "photos": names,
    "pixels": [packed(p) for p in pixels],
    "logits": {name: packed(t) for name, t in logits.items()},
    "top5": {name: t.topk(5).indices.tolist() for name, t in logits.items()},
}))
