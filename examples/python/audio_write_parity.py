"""Write twin (#83): soundfile reads a write-wav-produced file named by
RKTORCH_WAV_UNDER_TEST, so the writer is checked by an independent
decoder rather than this repo's own load-wav."""
import json
import os

import soundfile

data, rate = soundfile.read(os.environ["RKTORCH_WAV_UNDER_TEST"],
                            dtype="float32", always_2d=True)
print(json.dumps({
    "shape": [int(data.shape[1]), int(data.shape[0])],
    "rate": int(rate),
    "values": [float(v) for v in data.transpose().flatten().tolist()],
}))
