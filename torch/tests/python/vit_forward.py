"""VisionTransformer against its references, for vit-parity-test.rkt.

RKTORCH_VIT_TWIN names the safetensors file of a VisionTransformer at the
configuration of examples/python/18_dino.py; its ViT class, loaded from that
file, runs on a seeded [4 3 32 32] batch and on a [4 3 16 16] batch, the
second through its bicubic position interpolation. RKTORCH_VIT_TORCHVISION
names one at image 32, patch 4, width 192, depth 2, heads 3 and 10 classes,
which torchvision's VisionTransformer runs on the 32 px batch.
RKTORCH_VIT_WIDE names one at the twin's configuration but a 32 by 48 image,
which the twin's blocks run on a [1 3 16 24] batch, its positions resized by
Hugging Face's interpolate_pos_encoding call. The inputs
travel back with the outputs so the Racket side runs on exactly them, on
RKTORCH_PARITY_DEVICE.
"""
import importlib.util
import json
import os

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import VisionTransformer

from pretrained_common import load_safetensors, packed

EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "..", "examples", "python")


def dino():
    spec = importlib.util.spec_from_file_location(
        "dino", os.path.join(EXAMPLES, "18_dino.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qkv(ours, block, suffix):
    return torch.cat([ours[f"blocks.{block}.self-attn.{p}.{suffix}"]
                      for p in ("query", "key", "value")])


def twin_state(ours, depth):
    state = {"cls_token": ours["cls-token"], "pos_embed": ours["pos-embed"],
             "norm.weight": ours["norm.weight"], "norm.bias": ours["norm.bias"]}
    for suffix in ("weight", "bias"):
        state[f"patch.{suffix}"] = ours[f"patch-embed.proj.{suffix}"]
        for i in range(depth):
            block = f"blocks.{i}."
            state[f"{block}qkv.{suffix}"] = qkv(ours, i, suffix)
            for theirs, mine in (("norm1", "norm1"), ("norm2", "norm2"),
                                 ("proj", "self-attn.out"),
                                 ("fc1", "linear1"), ("fc2", "linear2")):
                state[f"{block}{theirs}.{suffix}"] = ours[f"{block}{mine}.{suffix}"]
    return state


def torchvision_state(ours, depth):
    state = {"class_token": ours["cls-token"],
             "encoder.pos_embedding": ours["pos-embed"]}
    for suffix in ("weight", "bias"):
        state[f"conv_proj.{suffix}"] = ours[f"patch-embed.proj.{suffix}"]
        state[f"encoder.ln.{suffix}"] = ours[f"norm.{suffix}"]
        state[f"heads.head.{suffix}"] = ours[f"head.{suffix}"]
        for i in range(depth):
            theirs = f"encoder.layers.encoder_layer_{i}."
            mine = f"blocks.{i}."
            state[f"{theirs}self_attention.in_proj_{suffix}"] = qkv(ours, i, suffix)
            for t, m in (("ln_1", "norm1"), ("ln_2", "norm2"),
                         ("self_attention.out_proj", "self-attn.out"),
                         ("mlp.0", "linear1"), ("mlp.3", "linear2")):
                state[f"{theirs}{t}.{suffix}"] = ours[f"{mine}{m}.{suffix}"]
    return state


def interpolate_pos_encoding(pos_embed, grid, height, width, patch):
    """transformers' ViTEmbeddings.interpolate_pos_encoding, from a grid that
    need not be square."""
    dim = pos_embed.shape[-1]
    patch_pos_embed = pos_embed[:, 1:].reshape(1, grid[0], grid[1], dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
    patch_pos_embed = F.interpolate(
        patch_pos_embed, size=(height // patch, width // patch),
        mode="bicubic", align_corners=False)
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((pos_embed[:, :1], patch_pos_embed), dim=1)


def wide_forward(vit, x, grid, patch):
    h = vit.patch(x).flatten(2).transpose(1, 2)
    h = torch.cat([vit.cls_token.expand(h.shape[0], -1, -1), h], 1)
    h = h + interpolate_pos_encoding(vit.pos_embed, grid, *x.shape[-2:], patch)
    for block in vit.blocks:
        h = block(h)
    return vit.norm(h)[:, 0]


device = os.environ.get("RKTORCH_PARITY_DEVICE", "cpu")
torch.manual_seed(0)
x32 = torch.randn(4, 3, 32, 32)
x16 = torch.randn(4, 3, 16, 16)
x_wide = torch.randn(1, 3, 16, 24)

module = dino()
twin = module.ViT()
twin.load_state_dict(twin_state(load_safetensors(os.environ["RKTORCH_VIT_TWIN"]),
                                module.DEPTH))
twin = twin.to(device).eval()

wide_grid = (32 // module.PATCH, 48 // module.PATCH)
wide = module.ViT()
wide.pos_embed = nn.Parameter(
    torch.zeros(1, 1 + wide_grid[0] * wide_grid[1], module.DIM))
wide.load_state_dict(twin_state(load_safetensors(os.environ["RKTORCH_VIT_WIDE"]),
                                module.DEPTH))
wide = wide.to(device).eval()

reference = VisionTransformer(image_size=32, patch_size=4, num_layers=2,
                              num_heads=3, hidden_dim=192, mlp_dim=768,
                              num_classes=10)
reference.load_state_dict(torchvision_state(
    load_safetensors(os.environ["RKTORCH_VIT_TORCHVISION"]), 2))
reference = reference.to(device).eval()

with torch.no_grad():
    outputs = {"twin32": twin(x32.to(device)), "twin16": twin(x16.to(device)),
               "torchvision": reference(x32.to(device)),
               "wide": wide_forward(wide, x_wide.to(device), wide_grid,
                                    module.PATCH)}

print(json.dumps({"x32": packed(x32), "x16": packed(x16),
                  "x_wide": packed(x_wide),
                  **{k: packed(v.cpu()) for k, v in outputs.items()}}))
