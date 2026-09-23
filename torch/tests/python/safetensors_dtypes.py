"""A safetensors file with every dtype the Racket container writes, built
by hand (header JSON plus the element bytes, as the format specifies), so
the Racket load-state! reads it and save-state! reproduces each payload.
"""
import json
import struct
import torch

torch.manual_seed(0)
entries = {
    "f32": torch.randn(2, 3),
    "f64": torch.randn(3).to(torch.float64),
    "f16": torch.randn(2, 2).to(torch.float16),
    "bf16": torch.randn(4).to(torch.bfloat16),
    "i64": torch.tensor([[1, -2], [3, 4]]),
    "bool": torch.tensor([True, False, True]),
    "u8": torch.tensor([0, 9, 255], dtype=torch.uint8),
}
tags = {
    torch.float32: "F32", torch.float64: "F64", torch.float16: "F16",
    torch.bfloat16: "BF16", torch.int64: "I64", torch.bool: "BOOL",
    torch.uint8: "U8",
}
header = {}
payload = b""
for name, t in entries.items():
    raw = t.contiguous().view(torch.uint8).numpy().tobytes() \
        if t.dtype != torch.bool else t.numpy().tobytes()
    header[name] = {
        "dtype": tags[t.dtype],
        "shape": list(t.shape),
        "data_offsets": [len(payload), len(payload) + len(raw)],
    }
    payload += raw
header_bytes = json.dumps(header).encode("utf-8")
blob = struct.pack("<Q", len(header_bytes)) + header_bytes + payload
print(json.dumps({
    "hex": blob.hex(),
    "values": {name: t.to(torch.float32).flatten().tolist()
               for name, t in entries.items()},
    "payload_hex": {name: (t.contiguous().view(torch.uint8).numpy().tobytes()
                           if t.dtype != torch.bool else t.numpy().tobytes()).hex()
                    for name, t in entries.items()},
}))
