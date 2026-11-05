"""copy.deepcopy of an nn.Module graph with cycles.

A Holder module registers a Leaf child and keeps a plain list that holds
itself; the Leaf keeps a plain list holding its parent, a back-pointer
outside the module registry. Each graph is printed as its identity shape:
nodes numbered in depth-first order over the fields named in FIELDS, each
with its kind and the numbers of the nodes its fields reach.
"""
import copy
import json

import torch
from torch import nn


class Leaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.up = [None]


class Holder(nn.Module):
    def __init__(self, kid):
        super().__init__()
        self.kid = kid
        self.loop = [None]
        self.loop[0] = self.loop


FIELDS = {Holder: ["kid", "loop"], Leaf: ["weight", "up"]}


def kind_of(x):
    if isinstance(x, nn.Module):
        return type(x).__name__.lower()
    if isinstance(x, torch.Tensor):
        return "tensor"
    return "list"


def parts_of(x):
    if isinstance(x, nn.Module):
        return [getattr(x, f) for f in FIELDS[type(x)]]
    if isinstance(x, torch.Tensor):
        return []
    return list(x)


def shape(root):
    ids = {}
    nodes = []

    def visit(x):
        if id(x) in ids:
            return ids[id(x)]
        ids[id(x)] = len(nodes)
        node = {"kind": kind_of(x), "edges": []}
        nodes.append(node)
        node["edges"] = [visit(p) for p in parts_of(x)]
        return ids[id(x)]

    visit(root)
    return nodes, set(ids)


h = Holder(Leaf())
h.kid.up[0] = h
c = copy.deepcopy(h)
original, original_ids = shape(h)
copied, copied_ids = shape(c)
print(json.dumps({
    "original": original,
    "copy": copied,
    "shared": len(original_ids & copied_ids),
}))
