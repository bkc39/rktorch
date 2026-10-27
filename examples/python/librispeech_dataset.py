"""LibriSpeech dataset twin (#265): torchaudio.datasets.LIBRISPEECH under the
root RKTORCH_LIBRISPEECH_ROOT names, against torch/audio/librispeech.rkt's
`librispeech` in torch/tests/librispeech-parity-test.rkt. The batch is
pad_sequence over the waveforms, the default collate's zero padding."""
import json
import os

import torchaudio
from torch.nn.utils.rnn import pad_sequence

root = os.environ["RKTORCH_LIBRISPEECH_ROOT"]
ds = torchaudio.datasets.LIBRISPEECH(root, url="dev-clean")
items = [ds[i] for i in range(len(ds))]


def values(t):
    return [float(v) for v in t.flatten().tolist()]


x = pad_sequence([w[0] for (w, *_rest) in items], batch_first=True).unsqueeze(1)
print(json.dumps({
    "length": len(ds),
    "ids": [f"{s}-{c}-{u:04d}" for (_w, _r, _t, s, c, u) in items],
    "transcripts": [t for (_w, _r, t, *_rest) in items],
    "items": [{"shape": list(w.shape), "values": values(w)}
              for (w, *_rest) in items],
    "x_shape": list(x.shape),
    "x_values": values(x),
    "lengths": [w.shape[-1] for (w, *_rest) in items],
}))
