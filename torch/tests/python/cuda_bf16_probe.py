"""Oracle of autocast-test.rkt's G3-L8 (T2 spec, #385): whether the current
CUDA device's compute capability major is at least 8, torch's
is_bf16_supported(including_emulation=False). Prints it as JSON."""

import json

import torch

print(json.dumps({"bf16": torch.cuda.is_available()
                  and torch.cuda.get_device_capability()[0] >= 8}))
