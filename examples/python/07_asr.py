"""Parity twin of examples/racket/07-asr.rkt: same seed, same hybrid
CTC/attention encoder-decoder in the same declaration order, 5 Adam steps
on the committed MISTER QUILTER fixture."""

import json
import math
import os

import torch
import torchaudio
from torch import nn

FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "torch", "audio",
    "fixtures", "librispeech-1272-128104-0000.flac")

DEVICE = os.environ.get("RKTORCH_PARITY_DEVICE") or "cpu"

N_MELS = 80
N_EMBD = 64
N_HEAD = 4
CTC_WEIGHT = 0.3


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


def sinusoidal_positions(t_len, n_embd):
    """The racket side's sin-half | cos-half layout, not interleaved."""
    half = n_embd // 2
    positions = torch.arange(t_len, dtype=torch.float32).unsqueeze(1)
    freqs = torch.exp(torch.arange(half, dtype=torch.float32)
                      * (-math.log(10000.0) / half))
    angles = positions * freqs.unsqueeze(0)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.wq = nn.Linear(N_EMBD, N_EMBD)
        self.wk = nn.Linear(N_EMBD, N_EMBD)
        self.wv = nn.Linear(N_EMBD, N_EMBD)
        self.wo = nn.Linear(N_EMBD, N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.fc1 = nn.Linear(N_EMBD, 4 * N_EMBD)
        self.fc2 = nn.Linear(4 * N_EMBD, N_EMBD)

    def forward(self, x):
        b, t, _ = x.shape
        hd = N_EMBD // N_HEAD

        def split(m):
            return m.reshape(b, t, N_HEAD, hd).transpose(1, 2)

        xn = self.ln1(x)
        q, k, v = split(self.wq(xn)), split(self.wk(xn)), split(self.wv(xn))
        att = torch.softmax(q @ k.transpose(2, 3) / math.sqrt(hd), dim=-1)
        ctx = (att @ v).transpose(1, 2).reshape(b, t, N_EMBD)
        x1 = x + self.wo(ctx)
        return x1 + self.fc2(nn.functional.gelu(self.fc1(self.ln2(x1))))


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.sq = nn.Linear(N_EMBD, N_EMBD)
        self.sk = nn.Linear(N_EMBD, N_EMBD)
        self.sv = nn.Linear(N_EMBD, N_EMBD)
        self.so = nn.Linear(N_EMBD, N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.cq = nn.Linear(N_EMBD, N_EMBD)
        self.ck = nn.Linear(N_EMBD, N_EMBD)
        self.cv = nn.Linear(N_EMBD, N_EMBD)
        self.co = nn.Linear(N_EMBD, N_EMBD)
        self.ln3 = nn.LayerNorm(N_EMBD)
        self.fc1 = nn.Linear(N_EMBD, 4 * N_EMBD)
        self.fc2 = nn.Linear(4 * N_EMBD, N_EMBD)

    def forward(self, x, memory):
        b, s, _ = x.shape
        m = memory.shape[1]
        hd = N_EMBD // N_HEAD

        def split(t, length):
            return t.reshape(b, length, N_HEAD, hd).transpose(1, 2)

        xn = self.ln1(x)
        q = split(self.sq(xn), s)
        k = split(self.sk(xn), s)
        v = split(self.sv(xn), s)
        scores = q @ k.transpose(2, 3) / math.sqrt(hd)
        causal = torch.tril(torch.ones(s, s, device=x.device)) == 0
        att = torch.softmax(scores.masked_fill(causal, -torch.inf), dim=-1)
        x1 = x + self.so((att @ v).transpose(1, 2).reshape(b, s, N_EMBD))
        x1n = self.ln2(x1)
        q2 = split(self.cq(x1n), s)
        k2 = split(self.ck(memory), m)
        v2 = split(self.cv(memory), m)
        att2 = torch.softmax(q2 @ k2.transpose(2, 3) / math.sqrt(hd), dim=-1)
        x2 = x1 + self.co((att2 @ v2).transpose(1, 2).reshape(b, s, N_EMBD))
        return x2 + self.fc2(nn.functional.gelu(self.fc1(self.ln3(x2))))


class ASR(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.conv1 = nn.Conv1d(N_MELS, N_EMBD, 3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(N_EMBD, N_EMBD, 3, stride=2, padding=1)
        self.dil1 = nn.Conv1d(N_EMBD, N_EMBD, 3, dilation=1, padding=1)
        self.dil2 = nn.Conv1d(N_EMBD, N_EMBD, 3, dilation=2, padding=2)
        self.dil3 = nn.Conv1d(N_EMBD, N_EMBD, 3, dilation=4, padding=4)
        self.enc1 = EncoderBlock()
        self.enc2 = EncoderBlock()
        self.ln_enc = nn.LayerNorm(N_EMBD)
        self.ctc_head = nn.Linear(N_EMBD, vocab_size + 1)
        self.tok_emb = nn.Embedding(vocab_size + 2, N_EMBD)
        self.dec1 = DecoderBlock()
        self.dec2 = DecoderBlock()
        self.ln_dec = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size + 1)

    def forward(self, x, dec_in):
        c = torch.relu(self.conv2(torch.relu(self.conv1(x))))
        c = c + torch.relu(self.dil1(c))
        c = c + torch.relu(self.dil2(c))
        c = c + torch.relu(self.dil3(c))
        t = c.shape[2]
        pos = sinusoidal_positions(t, N_EMBD).to(x.device)
        memory = self.ln_enc(self.enc2(self.enc1(c.transpose(1, 2) + pos)))
        ctc_log_probs = torch.log_softmax(self.ctc_head(memory), dim=2)
        s = dec_in.shape[1]
        dpos = sinusoidal_positions(s, N_EMBD).to(x.device)
        d = self.ln_dec(self.dec2(self.dec1(self.tok_emb(dec_in) + dpos,
                                            memory), memory))
        return ctc_log_probs, self.head(d)


waveform, rate = torchaudio.load(FIXTURE)
transcript = ("MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES "
              "AND WE ARE GLAD TO WELCOME HIS GOSPEL")
vocab = sorted(set(transcript))
char_to_id = {c: i for i, c in enumerate(vocab)}
vocab_size = len(vocab)
eos, sos = vocab_size, vocab_size + 1

features = log_mel(waveform[0], rate).unsqueeze(0)
ids = [char_to_id[c] for c in transcript]
targets = torch.tensor([ids], dtype=torch.int64)
dec_in = torch.tensor([[sos] + ids], dtype=torch.int64)
dec_out = torch.tensor([ids + [eos]], dtype=torch.int64)

torch.manual_seed(0)

# construct on DEVICE: the seeded init must draw from that device's generator
with torch.device(DEVICE):
    net = ASR(vocab_size)
features = features.to(DEVICE)
targets, dec_in, dec_out = (targets.to(DEVICE), dec_in.to(DEVICE),
                            dec_out.to(DEVICE))
opt = torch.optim.Adam(net.parameters(), lr=0.001)

losses = []
for _ in range(5):
    opt.zero_grad()
    ctc_lp, logits = net(features, dec_in)
    loss_ctc = nn.functional.ctc_loss(
        ctc_lp.transpose(0, 1), targets,
        input_lengths=torch.tensor([ctc_lp.shape[1]]),
        target_lengths=torch.tensor([len(transcript)]),
        blank=vocab_size, reduction="mean", zero_infinity=False)
    loss_ce = nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size + 1), dec_out.reshape(-1))
    loss = CTC_WEIGHT * loss_ctc + (1.0 - CTC_WEIGHT) * loss_ce
    loss.backward()
    opt.step()
    losses.append(loss.item())

params = torch.cat([p.detach().flatten() for p in net.parameters()])

print(json.dumps({
    "shape": list(params.shape),
    "values": params.tolist(),
    "losses": losses,
}))
