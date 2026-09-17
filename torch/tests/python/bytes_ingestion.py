"""uint8 tensors from byte buffers: torch.frombuffer parity for #58.

The reprs pin PyTorch's integer column padding and the dtype suffix it
prints for every dtype but its defaults; the scaled image checks that
uint8 -> float32 -> / 255 matches on all 256 byte values.
"""
import json
import torch

buf = bytearray([0, 1, 2, 255])
t = torch.frombuffer(buf, dtype=torch.uint8)
every = torch.frombuffer(bytearray(range(256)), dtype=torch.uint8)
big = torch.frombuffer(bytearray(bytes(range(256)) * 5), dtype=torch.uint8)
print(json.dumps({
    "values": t.tolist(),
    "dtype": str(t.dtype),
    "repr": repr(t),
    "repr_2x2": repr(t.reshape(2, 2)),
    "empty_repr": repr(torch.empty(0, dtype=torch.uint8)),
    "summarized_repr": repr(big),
    "scaled": (every.to(torch.float32) / 255.0).tolist(),
    "as_int64_dtype": str(t.to(torch.int64).dtype),
    "float64_repr": repr(torch.tensor([[1.5, -2.0], [0.0, 3.25]],
                                      dtype=torch.float64)),
    "item": torch.frombuffer(bytearray([7]), dtype=torch.uint8).item(),
}))
