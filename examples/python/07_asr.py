"""Parity twin of examples/racket/07-asr.rkt: same seed, same Conv1d/Conv1d/
Linear encoder in the same declaration order, 5 Adam steps of CTC loss on the
committed MISTER QUILTER fixture."""

import json
import os

import torch
import torchaudio
from torch import nn

FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "torch", "audio",
    "fixtures", "librispeech-1272-128104-0000.flac")

DEVICE = os.environ.get("RKTORCH_PARITY_DEVICE") or "cpu"

N_MELS = 80
N_HIDDEN = 64


def log_mel(x, rate):
    """torch/audio/functional.rkt's log-mel-spectrogram, spelled in torch."""
    window = torch.hann_window(400)
    frames = torch.view_as_real(
        torch.stft(x, n_fft=400, hop_length=160, window=window,
                   center=True, pad_mode="reflect", normalized=False,
                   onesided=True, return_complex=True))
    power = frames[..., 0] ** 2 + frames[..., 1] ** 2
    fbank = torchaudio.functional.melscale_fbanks(
        n_freqs=201, f_min=0.0, f_max=rate / 2.0, n_mels=N_MELS,
        sample_rate=rate, norm=None, mel_scale="htk")
    return torch.log(fbank.t() @ power + 1e-6)


class ASR(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.conv1 = nn.Conv1d(N_MELS, N_HIDDEN, 3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(N_HIDDEN, N_HIDDEN, 3, stride=2, padding=1)
        self.head = nn.Linear(N_HIDDEN, vocab_size + 1)

    def forward(self, x):
        frames = torch.relu(self.conv2(torch.relu(self.conv1(x))))
        return torch.log_softmax(self.head(frames.transpose(1, 2)), dim=2)


waveform, rate = torchaudio.load(FIXTURE)
transcript = ("MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES "
              "AND WE ARE GLAD TO WELCOME HIS GOSPEL")
vocab = sorted(set(transcript))
char_to_id = {c: i for i, c in enumerate(vocab)}
vocab_size = len(vocab)

features = log_mel(waveform[0], rate).unsqueeze(0)
targets = torch.tensor([[char_to_id[c] for c in transcript]],
                       dtype=torch.int64)

torch.manual_seed(0)

# construct on DEVICE: the seeded init must draw from that device's generator
with torch.device(DEVICE):
    net = ASR(vocab_size)
features, targets = features.to(DEVICE), targets.to(DEVICE)
opt = torch.optim.Adam(net.parameters(), lr=0.001)

losses = []
for _ in range(5):
    opt.zero_grad()
    out = net(features)
    loss = nn.functional.ctc_loss(
        out.transpose(0, 1), targets,
        input_lengths=torch.tensor([out.shape[1]]),
        target_lengths=torch.tensor([len(transcript)]),
        blank=vocab_size, reduction="mean", zero_infinity=False)
    loss.backward()
    opt.step()
    losses.append(loss.item())

params = torch.cat([p.detach().flatten() for p in net.parameters()])

print(json.dumps({
    "shape": list(params.shape),
    "values": params.tolist(),
    "losses": losses,
}))
