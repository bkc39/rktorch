"""Spectral twin (#83): torch.stft + torchaudio.functional.melscale_fbanks
on the committed speech fixture, compared in
torch/tests/audio-parity-test.rkt."""
import json
import os

import torch
import torchaudio

path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "torch", "audio", "fixtures",
                    "librispeech-1272-128104-0000.flac")
waveform, rate = torchaudio.load(path)
x = waveform[0]

window = torch.hann_window(400)
frames = torch.view_as_real(
    torch.stft(x, n_fft=400, hop_length=160, window=window,
               center=True, pad_mode="reflect", normalized=False,
               onesided=True, return_complex=True)).contiguous()

fbank = torchaudio.functional.melscale_fbanks(
    n_freqs=201, f_min=0.0, f_max=rate / 2.0, n_mels=80,
    sample_rate=rate, norm=None, mel_scale="htk")

power = frames[..., 0] ** 2 + frames[..., 1] ** 2
log_mel = torch.log(fbank.t() @ power + 1e-6)

print(json.dumps({
    "stft_shape": list(frames.shape),
    "stft_head": [float(v) for v in frames.flatten()[:64].tolist()],
    "fbank_shape": list(fbank.shape),
    "fbank": [float(v) for v in fbank.flatten().tolist()],
    "log_mel_shape": list(log_mel.shape),
    "log_mel_head": [float(v) for v in log_mel.flatten()[:64].tolist()],
}))
