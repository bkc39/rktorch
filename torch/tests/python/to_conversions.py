"""Tensor.to and Module.to on the CPU path (#15's device-op parity gap).

The dtype casts pin repr parity; the identity checks pin that a no-op
`.to` returns the very same object, which the Racket `to` mirrors.
"""
import json
import torch
import torch.nn as nn

x = torch.tensor([[1.5, -2.0], [0.0, 3.25]])
torch.manual_seed(0)
lin = nn.Linear(2, 2)


class Counted(nn.Module):
    """a submodule plus an int64 counter and a bool mask as buffers"""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)
        self.register_buffer("steps", torch.tensor([0, 1]))
        self.register_buffer("keep", torch.tensor([True, False]))

    def forward(self, x):
        return self.lin(x)


counted = Counted().to(torch.float64)
print(json.dumps({
    "float64_values": x.to(torch.float64).flatten().tolist(),
    "float64_dtype": str(x.to(torch.float64).dtype),
    "int64_repr": repr(x.to(torch.int64)),
    "bool_repr": repr(x.to(torch.bool)),
    "both_dtype": str(x.to("cpu", torch.float64).dtype),
    "zeros_int64_repr": repr(torch.zeros((2, 3), dtype=torch.int64)),
    "full_int64_repr": repr(torch.full((2,), 7, dtype=torch.int64)),
    "ones_bool_repr": repr(torch.ones(3, dtype=torch.bool)),
    "zeros_like_dtype": str(torch.zeros_like(x.to(torch.float64)).dtype),
    "cpu_is_self": x.to("cpu") is x,
    "dtype_is_self": x.to(torch.float32) is x,
    "device_type": x.device.type,
    "cuda_available": torch.cuda.is_available(),
    "linear_is_self": lin.to(torch.float64) is lin,
    "linear_state_dtypes": [str(v.dtype) for v in lin.state_dict().values()],
    "counted_dtypes": {k: str(v.dtype) for k, v in counted.state_dict().items()},
    "linear_state_values": [float(v) for v in
                            torch.cat([p.detach().flatten()
                                       for p in lin.parameters()]).tolist()],
}))
