"""The causal-attention mask idiom, end to end.

Build the mask from tril + ==, fill the upper triangle with -inf, and
softmax — exactly what the 06-gpt capstone's attention will do. The
recipe battery can't express -inf, so this composition is checked here.
"""
import json
import torch

torch.manual_seed(0)
scores = torch.randn(4, 4)
mask = torch.tril(torch.ones(4, 4)) == 0
r = torch.softmax(scores.masked_fill(mask, float("-inf")), -1)
print(json.dumps({
    "shape": list(r.shape),
    "values": [float(v) for v in r.flatten().tolist()],
}))
