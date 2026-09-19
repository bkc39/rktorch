"""torchvision.utils.make_grid on a seeded batch, and save_image's
quantization of it, for the Racket image-grid and write-ppm pair.

The CUDA parity shell carries torch-bin without torchvision, so the grid
falls back to make_grid's own algorithm (torchvision/utils.py) there.
"""
import json
import math
import torch

try:
    from torchvision.utils import make_grid
except ImportError:
    def make_grid(tensor, nrow=8, padding=2, pad_value=0.0):
        if tensor.size(1) == 1:
            tensor = torch.cat((tensor, tensor, tensor), 1)
        if tensor.size(0) == 1:
            return tensor.squeeze(0)
        nmaps = tensor.size(0)
        xmaps = min(nrow, nmaps)
        ymaps = int(math.ceil(float(nmaps) / xmaps))
        height, width = tensor.size(2) + padding, tensor.size(3) + padding
        grid = tensor.new_full(
            (tensor.size(1), height * ymaps + padding, width * xmaps + padding),
            pad_value)
        k = 0
        for y in range(ymaps):
            for x in range(xmaps):
                if k >= nmaps:
                    break
                grid.narrow(1, y * height + padding, height - padding).narrow(
                    2, x * width + padding, width - padding).copy_(tensor[k])
                k = k + 1
        return grid

torch.manual_seed(0)
x = torch.rand(5, 3, 4, 4)
g = make_grid(x, nrow=2, padding=1, pad_value=0.5)
q = g.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8).permute(1, 2, 0)
torch.manual_seed(1)
one = make_grid(torch.rand(1, 3, 4, 4), nrow=2, padding=1, pad_value=0.5)
print(json.dumps({
    "shape": list(g.shape),
    "values": [float(v) for v in g.flatten().tolist()],
    "pixels": [int(v) for v in q.contiguous().flatten().tolist()],
    "one_shape": list(one.shape),
    "one_values": [float(v) for v in one.flatten().tolist()],
}))
