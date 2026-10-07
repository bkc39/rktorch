"""Parity twin of examples/racket/07-asr.rkt: same seed, same hybrid
CTC/attention encoder-decoder in the same declaration order, 5 Adam steps
on the committed MISTER QUILTER fixture.

The stacks are the standard nn.TransformerEncoder and nn.TransformerDecoder
over pre-norm nn.TransformerEncoderLayer and nn.TransformerDecoderLayer
with an exact-gelu feed-forward four times the width and no dropout, each
ending in a LayerNorm, as the Racket TransformerEncoder and
TransformerDecoder with #:norm? #t build them: every block a copy of the
first. PyTorch's tgt_is_causal is only a hint that the mask beside it is
the causal mask, so the forward passes both; the Racket #:tgt-causal?
builds the mask itself.

The parameters are reported in the Racket model's order: each attention's
fused in_proj_weight and in_proj_bias split by rows into query, key and
value, each projection's weight followed by its bias.
"""

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
N_LAYER = 6
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
    """sinusoidal-positions #:layout 'halves: sines, then cosines."""
    half = n_embd // 2
    positions = torch.arange(t_len, dtype=torch.float32).unsqueeze(1)
    freqs = torch.exp(torch.arange(half, dtype=torch.float32)
                      * (-math.log(10000.0) / half))
    angles = positions * freqs.unsqueeze(0)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


def block_settings():
    return dict(dim_feedforward=4 * N_EMBD, dropout=0.0, activation="gelu",
                norm_first=True, batch_first=True)


class ASR(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.conv1 = nn.Conv1d(N_MELS, N_EMBD, 3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(N_EMBD, N_EMBD, 3, stride=2, padding=1)
        self.dils = nn.ModuleList([
            nn.Conv1d(N_EMBD, N_EMBD, 3, dilation=d, padding=d)
            for d in (1, 2, 4, 8)])
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(N_EMBD, N_HEAD, **block_settings()),
            N_LAYER, norm=nn.LayerNorm(N_EMBD), enable_nested_tensor=False)
        self.ctc_head = nn.Linear(N_EMBD, vocab_size + 1)
        self.tok_emb = nn.Embedding(vocab_size + 2, N_EMBD)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(N_EMBD, N_HEAD, **block_settings()),
            N_LAYER, norm=nn.LayerNorm(N_EMBD))
        self.head = nn.Linear(N_EMBD, vocab_size + 1)

    def forward(self, x, dec_in, lengths=None):
        def halve(n):
            return (n + 1) // 2

        t0 = x.shape[2]
        t1, t2 = halve(t0), halve(halve(t0))
        l1 = [halve(n) for n in lengths] if lengths else None
        l2 = [halve(n) for n in l1] if l1 else None

        def clip(v, lens, t):
            # re-zero padding after every biased convolution, else the next
            # kernel blends pad activations into real boundary frames
            if not lens:
                return v
            idx = torch.arange(t, device=v.device).unsqueeze(0)
            keep = (idx < torch.tensor(lens, device=v.device,
                                       dtype=torch.float32).unsqueeze(1))
            return v * keep.to(v.dtype).unsqueeze(1)

        c = clip(torch.relu(self.conv1(x)), l1, t1)
        c = clip(torch.relu(self.conv2(c)), l2, t2)
        for dil in self.dils:
            c = clip(c + torch.relu(dil(c)), l2, t2)
        padding = None
        if l2:
            idx = torch.arange(t2, device=x.device).unsqueeze(0)
            padding = idx >= torch.tensor(l2, device=x.device,
                                          dtype=torch.float32).unsqueeze(1)
        e = c.transpose(1, 2) + sinusoidal_positions(
            c.shape[2], N_EMBD).to(x.device)
        memory = self.encoder(e, src_key_padding_mask=padding)
        ctc_log_probs = torch.log_softmax(self.ctc_head(memory), dim=2)
        s = dec_in.shape[1]
        causal = nn.Transformer.generate_square_subsequent_mask(
            s, device=x.device)
        d = self.tok_emb(dec_in) + sinusoidal_positions(
            s, N_EMBD).to(x.device)
        d = self.decoder(d, memory, tgt_mask=causal, tgt_is_causal=True,
                         memory_key_padding_mask=padding)
        return ctc_log_probs, self.head(d)


def racket_order(net):
    for name, p in net.named_parameters():
        if name.endswith("in_proj_weight"):
            attention = net.get_submodule(name.rsplit(".", 1)[0])
            for weight, bias in zip(attention.in_proj_weight.chunk(3),
                                    attention.in_proj_bias.chunk(3)):
                yield weight
                yield bias
        elif not name.endswith("in_proj_bias"):
            yield p


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
        blank=vocab_size, reduction="mean", zero_infinity=True)
    loss_ce = nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size + 1), dec_out.reshape(-1))
    loss = CTC_WEIGHT * loss_ctc + (1.0 - CTC_WEIGHT) * loss_ce
    loss.backward()
    opt.step()
    losses.append(loss.item())

params = torch.cat([p.detach().flatten() for p in racket_order(net)])

print(json.dumps({
    "shape": list(params.shape),
    "values": params.tolist(),
    "losses": losses,
}))
