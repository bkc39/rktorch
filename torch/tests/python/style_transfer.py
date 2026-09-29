"""torchvision's VGG-16 features and five steps of the style transfer in
examples/racket/16-style-transfer.rkt, for style-transfer-parity-test.rkt.
The weights come from the features-only safetensors file the Racket side
loads (RKTORCH_PARITY_WEIGHTS names its directory). The content and style
images travel back as the pixels used here, so both sides start from the
same tensors: the activations at the five style layers, then each step's
two losses and the final image.
"""
import json
import os

import torch
import torchvision.transforms.functional as F
from torchvision import models
from torchvision.io import ImageReadMode, decode_image, read_file

from pretrained_common import packed, weights

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "vision", "fixtures")
STYLE_LAYERS = [0, 2, 5, 7, 10]
CONTENT_LAYER = 7
SIZE, STEPS, LR = 64, 5, 0.02


def image(*path):
    pixels = decode_image(read_file(os.path.join(FIXTURES, *path)),
                          ImageReadMode.RGB)
    return F.resize(F.convert_image_dtype(pixels), SIZE).unsqueeze(0)


def activations(net, x, layers):
    x = F.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    found = {}
    for i, step in enumerate(net):
        if i > max(layers):
            break
        x = step(x)
        if i in layers:
            found[i] = x
    return found


def gram(f):
    n, c, h, w = f.shape
    m = f.reshape(n * c, h * w)
    return m @ m.t() / (n * c * h * w)


net = models.vgg16().features
net.load_state_dict({k.removeprefix("features."): v
                     for k, v in weights("vgg16-features").items()})
# torchvision's ReLUs are in place and would overwrite the convolution
# outputs kept above them, as the PyTorch tutorial also notes
for i, m in enumerate(net):
    if isinstance(m, torch.nn.ReLU):
        net[i] = torch.nn.ReLU(inplace=False)
for p in net.parameters():
    p.requires_grad_(False)

content = image("hymenoptera", "bees", "honey-bee.jpg")
style = image("style", "starry-night.jpg")

with torch.no_grad():
    found = activations(net, content, STYLE_LAYERS)
    style_grams = {i: gram(f) for i, f in
                   activations(net, style, STYLE_LAYERS).items()}
    target = activations(net, content, [CONTENT_LAYER])[CONTENT_LAYER]

picture = content.clone().requires_grad_(True)
opt = torch.optim.Adam([picture], lr=LR)
losses = []
for step in range(1, STEPS + 1):
    opt.zero_grad()
    feats = activations(net, picture, [CONTENT_LAYER] + STYLE_LAYERS)
    style_loss = sum(torch.nn.functional.mse_loss(gram(feats[i]), style_grams[i])
                     for i in STYLE_LAYERS)
    content_loss = torch.nn.functional.mse_loss(feats[CONTENT_LAYER], target)
    (style_loss * 1e6 + content_loss).backward()
    opt.step()
    with torch.no_grad():
        picture.clamp_(0.0, 1.0)
    losses.append([step, style_loss.item(), content_loss.item()])

print(json.dumps({
    "content": packed(content),
    "style": packed(style),
    "activations": {str(i): packed(f) for i, f in found.items()},
    "losses": losses,
    "image": packed(picture),
}))
