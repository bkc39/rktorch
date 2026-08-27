"""Audio twin (#83): torchaudio.load on the committed fixture, compared
bit-for-bit against load-wav in torch/tests/python-cross-test.rkt."""
import json
import os

import torchaudio

path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "torch", "audio", "fixtures",
                    "sine-440-16k.wav")
waveform, rate = torchaudio.load(path)
print(json.dumps({
    "shape": list(waveform.shape),
    "rate": rate,
    "values": [float(v) for v in waveform.flatten().tolist()],
}))
