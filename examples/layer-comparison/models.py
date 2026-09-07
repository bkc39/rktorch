import math

import torch
from torch import nn
from torch.nn import functional as F


class Projection(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        self.bias = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound))

    def forward(self, x):
        return x @ self.weight.T + self.bias


class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.norm1 = ChannelNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = ChannelNorm(out_channels)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, stride=stride)
            if stride != 1 or in_channels != out_channels else None
        )

    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut is not None else x
        h = F.relu(self.norm1(self.conv1(x)))
        return F.relu(self.norm2(self.conv2(h)) + residual)


class ResidualStage(nn.Module):
    def __init__(self, in_channels, out_channels, depth, *, stride=1):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(in_channels if i == 0 else out_channels, out_channels,
                          stride=stride if i == 0 else 1)
            for i in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SmallResNet(nn.Module):
    def __init__(self, *, classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.stage1 = ResidualStage(16, 16, 2)
        self.stage2 = ResidualStage(16, 32, 2, stride=2)
        self.stage3 = ResidualStage(32, 64, 2, stride=2)
        self.head = Projection(64, classes)

    def forward(self, images):
        h = self.stage3(self.stage2(self.stage1(F.relu(self.stem(images)))))
        return self.head(h.mean(dim=(2, 3)))


class CausalSelfAttention(nn.Module):
    def __init__(self, width, heads, max_t, *, dropout=0.1):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("width must be positive and divisible by heads")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.q = Projection(width, width)
        self.k = Projection(width, width)
        self.v = Projection(width, width)
        self.out = Projection(width, width)
        self.register_buffer("mask", torch.ones(max_t, max_t).tril() == 0)
        self.attention_drop = nn.Dropout(dropout)
        self.output_drop = nn.Dropout(dropout)

    def forward(self, x):
        batch, time, _ = x.shape

        def split_heads(projection):
            return projection(x).reshape(batch, time, self.heads, self.head_dim).transpose(1, 2)

        queries, keys, values = split_heads(self.q), split_heads(self.k), split_heads(self.v)
        scores = (queries @ keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        active_mask = self.mask[:time, :time]
        weights = self.attention_drop(scores.masked_fill(active_mask, -math.inf).softmax(-1))
        joined = (weights @ values).transpose(1, 2).reshape(batch, time, self.width)
        return self.output_drop(self.out(joined))


class FeedForward(nn.Module):
    def __init__(self, width, *, dropout=0.1):
        super().__init__()
        self.up = Projection(width, 4 * width)
        self.down = Projection(4 * width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(F.gelu(self.up(x))))


class TransformerBlock(nn.Module):
    def __init__(self, width, heads, max_t, *, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = CausalSelfAttention(width, heads, max_t, dropout=dropout)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = FeedForward(width, dropout=dropout)

    def forward(self, x):
        h = x + self.attention(self.norm1(x))
        return h + self.mlp(self.norm2(h))


class TransformerStack(nn.Module):
    def __init__(self, width, heads, depth, max_t, *, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(width, heads, max_t, dropout=dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens):
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


if __name__ == "__main__":
    for model, x in [(SmallResNet(), torch.randn(2, 3, 32, 32)),
                     (TransformerStack(32, 4, 2, 16), torch.randn(2, 8, 32))]:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = output.square().mean()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            print(type(model).__name__, tuple(model(x).shape),
                  sum(p.numel() for p in model.parameters()), "parameters")
