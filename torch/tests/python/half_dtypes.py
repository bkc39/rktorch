"""The half pair: float16 and bfloat16 casts of a seeded tensor read back
through float32, their reprs, the constructors with the half dtypes, and a
layer moved to bfloat16 with its integer buffer left alone.
"""
import json
import torch
import torch.nn as nn

torch.manual_seed(0)
x = torch.randn(2, 3)
half = x.to(torch.float16)
brain = x.to(torch.bfloat16)


class Counted(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 2)
        self.register_buffer("steps", torch.tensor([0, 1]))


torch.manual_seed(1)
counted = Counted().to(torch.bfloat16)
print(json.dumps({
    "half_values": half.to(torch.float32).flatten().tolist(),
    "brain_values": brain.to(torch.float32).flatten().tolist(),
    "half_repr": repr(half),
    "brain_repr": repr(brain),
    "empty_half_repr": repr(torch.zeros(0, dtype=torch.float16)),
    "zeros_brain_repr": repr(torch.zeros(2, 2, dtype=torch.bfloat16)),
    "full_half_values": torch.full((3,), 0.1, dtype=torch.float16)
        .to(torch.float32).tolist(),
    "arange_brain_values": torch.arange(0, 5, dtype=torch.bfloat16)
        .to(torch.float32).tolist(),
    "tensor_half_values": torch.tensor([1.0, 0.1, 65504.0], dtype=torch.float16)
        .to(torch.float32).tolist(),
    "counted_dtypes": {k: str(v.dtype) for k, v in counted.state_dict().items()},
    "counted_weight": counted.lin.weight.to(torch.float32).flatten().tolist(),
}))
