"""nn.BatchNorm2d's num_batches_tracked past float32's last integer: the
state dict after one training forward from 2**24, as a safetensors file
built by hand, and the counter and its payload after one more training
forward and an eval forward."""
import json
import struct

import torch
from torch import nn

TAGS = {torch.float32: "F32", torch.int64: "I64"}


def safetensors(state):
    header = {}
    payload = b""
    for name, t in state.items():
        raw = t.contiguous().numpy().tobytes()
        header[name] = {
            "dtype": TAGS[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload += raw
    text = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(text)) + text + payload


torch.manual_seed(0)
m = nn.BatchNorm2d(3)
x = torch.randn(2, 3, 4, 4)
with torch.no_grad():
    m.num_batches_tracked.fill_(2 ** 24)
m(x)
saved = {k: v.detach().clone() for k, v in m.state_dict().items()}
m(x)
m.eval()
m(x)
counter = m.num_batches_tracked
print(json.dumps({
    "file": list(safetensors(saved)),
    "keys": list(saved),
    "dtypes": [str(v.dtype) for v in saved.values()],
    "saved_counter": int(saved["num_batches_tracked"]),
    "counter": int(counter),
    "counter_dtype": str(counter.dtype),
    "counter_bytes": list(counter.numpy().tobytes()),
}))
