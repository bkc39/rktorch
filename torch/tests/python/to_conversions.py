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
print(json.dumps({
    "float64_values": x.to(torch.float64).flatten().tolist(),
    "float64_dtype": str(x.to(torch.float64).dtype),
    "int64_repr": repr(x.to(torch.int64)),
    "bool_repr": repr(x.to(torch.bool)),
    "both_dtype": str(x.to("cpu", torch.float64).dtype),
    "cpu_is_self": x.to("cpu") is x,
    "dtype_is_self": x.to(torch.float32) is x,
    "device_type": x.device.type,
    "cuda_available": torch.cuda.is_available(),
    "linear_is_self": lin.to(torch.float64) is lin,
    "linear_state_dtypes": [str(v.dtype) for v in lin.state_dict().values()],
    "linear_state_values": [float(v) for v in
                            torch.cat([p.detach().flatten()
                                       for p in lin.parameters()]).tolist()],
}))
