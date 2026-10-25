"""Recipe of record for the trainer arc's capstone (#142, #383): DINO
self-distillation of a ViT-Tiny on CIFAR-10, in plain PyTorch, with the probe
protocol the Racket trainer must reproduce.

Pretraining sees all 50k training images and none of their labels. Multi-crop
runs on the device, batched: each view is one affine_grid + grid_sample over
the uint8 batch, then per-sample colour jitter (brightness, contrast,
saturation, hue, in that fixed order; torchvision draws the order per call),
grayscale and, on the second global view only, solarize. The random-resized
crop clips an out-of-range side to the image instead of torchvision's
ten-try rejection. Gradient clipping is on the global norm (DINO clips each
parameter). The run uses bf16 autocast unless --precision fp32.

Probes read the teacher backbone's CLS token in eval mode, from the centre
32 px image with no augmentation. Probe-fit and probe-select are a stratified
45k/5k split of the training set under SPLIT_SEED; both probes read features
standardised with probe-fit statistics. The linear probe is multinomial
logistic regression fitted by full-batch L-BFGS, its weight decay chosen on
probe-select; kNN is cosine, weighted by exp(sim / T), fitted on probe-fit
(leave-one-out when scored on probe-fit itself). The test split is read once,
after training, for the trained teacher and for both floors: a random-init
backbone and raw standardised pixels.

Outputs go to --out: config.json, metrics.jsonl (one line per epoch),
probe.json and teacher.pt.
"""

import argparse
import copy
import json
import math
import os
import tarfile
import time

import torch
import torch.nn.functional as F
from torch import nn

IMAGE = 32  # global view side, px
LOCAL = 16  # local view side, px
PATCH = 4  # patch side: 64 tokens + CLS on a global view, 16 + CLS local
DIM = 192  # ViT-Tiny width
DEPTH = 9  # transformer blocks
HEADS = 3  # attention heads
MLP_RATIO = 4  # block MLP hidden = MLP_RATIO * DIM

HEAD_HIDDEN = 1024  # DINO head MLP width
BOTTLENECK = 256  # DINO head output before the prototypes
PROTOTYPES = 4096  # K, the weight-normed last layer's outputs

GLOBAL_CROPS = 2  # 32 px views, the teacher's inputs
LOCAL_CROPS = 6  # 16 px views, student only
GLOBAL_SCALE = (0.4, 1.0)  # random-resized-crop area fraction, global
LOCAL_SCALE = (0.05, 0.4)  # random-resized-crop area fraction, local
RATIO = (3 / 4, 4 / 3)  # random-resized-crop aspect range
JITTER = (0.4, 0.4, 0.2, 0.1)  # brightness, contrast, saturation, hue
JITTER_P = 0.8  # probability a view is colour-jittered
GRAY_P = 0.2  # probability a view is grayscaled
SOLARIZE_P = 0.2  # probability the second global view is solarized
MEAN = (0.4914, 0.4822, 0.4465)  # CIFAR-10 channel means
STD = (0.2470, 0.2435, 0.2616)  # CIFAR-10 channel stds

BATCH = 256  # images per step; a 50k epoch is 195 steps, last partial dropped
BASE_LR = 5e-4  # AdamW lr at batch 256, scaled linearly with batch
WARMUP_EPOCHS = 10  # linear lr warm-up from 0
MIN_LR = 1e-6  # cosine lr floor
WD = (0.04, 0.4)  # weight decay, cosine; not on biases, norms, pos-embed, CLS
MOMENTUM = (0.996, 1.0)  # teacher EMA momentum, cosine
TEACHER_TEMP = (0.04, 0.07)  # teacher temperature, linear over the warm-up
TEACHER_TEMP_EPOCHS = 30  # teacher temperature warm-up length
STUDENT_TEMP = 0.1  # student temperature
CENTER_MOMENTUM = 0.9  # EMA of the teacher's batch-mean output
FREEZE_LAST_EPOCHS = 1  # epochs the head's last layer gets no gradient
CLIP = 3.0  # global gradient-norm clip

SPLIT_SEED = 0  # probe-fit / probe-select split, independent of --seed
SELECT_PER_CLASS = 500  # probe-select images per class (5k of 50k)
PROBE_WDS = tuple(10.0 ** k for k in range(-7, 0))  # linear-probe L2 grid, 1e-7 .. 1e-1
PROBE_ITERS = 1000  # L-BFGS iteration cap per fit
KNN_K = 20  # kNN neighbours
KNN_T = 0.07  # kNN vote temperature
KNN_EVERY = 20  # epochs between the cheap kNN-on-probe-select checks
FEATURE_BATCH = 1024  # images per no-grad feature pass

ARCHIVE = "cifar-10-binary.tar.gz"
RECORD = 3073


def load_cifar10(root):
    """Train and test as uint8 (N, 3, 32, 32) and int64 labels, from rktorch's
    cached binary archive when present, else torchvision's download."""
    path = os.path.join(root, ARCHIVE)
    if os.path.exists(path):
        with tarfile.open(path, "r:gz") as tar:
            blobs = {os.path.basename(m.name): tar.extractfile(m).read()
                     for m in tar.getmembers() if m.name.endswith(".bin")}

        def split(names):
            raw = torch.frombuffer(bytearray(b"".join(blobs[n] for n in names)),
                                   dtype=torch.uint8).reshape(-1, RECORD)
            return raw[:, 1:].reshape(-1, 3, IMAGE, IMAGE).clone(), raw[:, 0].long()

        train = split([f"data_batch_{i}.bin" for i in range(1, 6)])
        return train, split(["test_batch.bin"])
    import torchvision

    def tv(train):
        ds = torchvision.datasets.CIFAR10(root, train=train, download=True)
        return (torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous(),
                torch.tensor(ds.targets))

    return tv(True), tv(False)


def gray(x):
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)


def rgb_to_hsv(x):
    r, g, b = x.unbind(1)
    maxc, minc = x.max(1).values, x.min(1).values
    flat = maxc == minc
    ones = torch.ones_like(maxc)
    span = maxc - minc
    s = span / torch.where(flat, ones, maxc)
    div = torch.where(flat, ones, span)
    rc, gc, bc = (maxc - r) / div, (maxc - g) / div, (maxc - b) / div
    h = ((maxc == r) * (bc - gc)
         + ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
         + ((maxc != g) & (maxc != r)) * (4.0 + gc - rc))
    return torch.fmod(h / 6.0 + 1.0, 1.0), s, maxc


def hsv_to_rgb(h, s, v):
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    i = i.long() % 6
    p = (v * (1.0 - s)).clamp(0, 1)
    q = (v * (1.0 - s * f)).clamp(0, 1)
    t = (v * (1.0 - s * (1.0 - f))).clamp(0, 1)
    sector = (i.unsqueeze(1) == torch.arange(6, device=h.device).view(1, 6, 1, 1)).to(v.dtype)
    planes = [torch.stack(c, 1) for c in
              ((v, q, p, p, t, v), (t, v, v, q, p, p), (p, p, t, v, v, q))]
    return torch.stack([(sector * a).sum(1) for a in planes], 1)


def uniform(n, lo, hi, gen, device):
    return lo + (hi - lo) * torch.rand(n, device=device, generator=gen)


def coin(n, p, gen, device):
    return torch.rand(n, device=device, generator=gen) < p


def resized_crop(x, side, scale, gen):
    n, device = x.shape[0], x.device
    area = uniform(n, *scale, gen, device)
    ratio = torch.exp(uniform(n, math.log(RATIO[0]), math.log(RATIO[1]), gen, device))
    w = torch.sqrt(area * ratio).clamp(max=1.0)
    h = torch.sqrt(area / ratio).clamp(max=1.0)
    cx = (1 - w) * uniform(n, -1.0, 1.0, gen, device)
    cy = (1 - h) * uniform(n, -1.0, 1.0, gen, device)
    flip = torch.where(coin(n, 0.5, gen, device), -1.0, 1.0)
    zero = torch.zeros_like(w)
    theta = torch.stack([torch.stack([w * flip, zero, cx], 1),
                         torch.stack([zero, h, cy], 1)], 1)
    grid = F.affine_grid(theta, [n, 3, side, side], align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                         align_corners=False)


def colour_jitter(x, gen):
    n, device = x.shape[0], x.device
    on = coin(n, JITTER_P, gen, device)

    def factor(spread):
        f = uniform(n, 1 - spread, 1 + spread, gen, device)
        return torch.where(on, f, torch.ones_like(f)).view(n, 1, 1, 1)

    b, c, s = factor(JITTER[0]), factor(JITTER[1]), factor(JITTER[2])
    hue = torch.where(on, uniform(n, -JITTER[3], JITTER[3], gen, device),
                      torch.zeros(n, device=device)).view(n, 1, 1)
    x = (x * b).clamp(0, 1)
    m = gray(x).mean((1, 2, 3), keepdim=True)
    x = (c * x + (1 - c) * m).clamp(0, 1)
    x = (s * x + (1 - s) * gray(x)).clamp(0, 1)
    h, sat, v = rgb_to_hsv(x)
    return hsv_to_rgb(torch.fmod(h + hue + 1.0, 1.0), sat, v)


def view(images, side, scale, solarize_p, gen, mean, std):
    n, device = images.shape[0], images.device
    x = resized_crop(images.float() / 255.0, side, scale, gen)
    x = colour_jitter(x, gen)
    x = torch.where(coin(n, GRAY_P, gen, device).view(n, 1, 1, 1),
                    gray(x).expand_as(x), x)
    if solarize_p > 0:
        x = torch.where(coin(n, solarize_p, gen, device).view(n, 1, 1, 1)
                        & (x >= 0.5), 1.0 - x, x)
    return (x - mean) / std


def multi_crop(images, gen, mean, std):
    globals_ = [view(images, IMAGE, GLOBAL_SCALE, SOLARIZE_P if i == 1 else 0.0,
                     gen, mean, std) for i in range(GLOBAL_CROPS)]
    locals_ = [view(images, LOCAL, LOCAL_SCALE, 0.0, gen, mean, std)
               for _ in range(LOCAL_CROPS)]
    return torch.cat(globals_), torch.cat(locals_)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM, eps=1e-6)
        self.qkv = nn.Linear(DIM, 3 * DIM)
        self.proj = nn.Linear(DIM, DIM)
        self.norm2 = nn.LayerNorm(DIM, eps=1e-6)
        self.fc1 = nn.Linear(DIM, MLP_RATIO * DIM)
        self.fc2 = nn.Linear(MLP_RATIO * DIM, DIM)

    def forward(self, x):
        n, t, _ = x.shape
        q, k, v = (self.qkv(self.norm1(x)).reshape(n, t, 3, HEADS, DIM // HEADS)
                   .permute(2, 0, 3, 1, 4))
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(a.transpose(1, 2).reshape(n, t, DIM))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        grid = IMAGE // PATCH
        self.patch = nn.Conv2d(3, DIM, PATCH, stride=PATCH)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + grid * grid, DIM))
        self.blocks = nn.ModuleList(Block() for _ in range(DEPTH))
        self.norm = nn.LayerNorm(DIM, eps=1e-6)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def pos(self, side):
        grid, full = side // PATCH, IMAGE // PATCH
        if grid == full:
            return self.pos_embed
        cls, patches = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        patches = F.interpolate(
            patches.reshape(1, full, full, DIM).permute(0, 3, 1, 2),
            size=(grid, grid), mode="bicubic", align_corners=False)
        return torch.cat([cls, patches.permute(0, 2, 3, 1).reshape(1, -1, DIM)], 1)

    def forward(self, x):
        h = self.patch(x).flatten(2).transpose(1, 2)
        h = torch.cat([self.cls_token.expand(h.shape[0], -1, -1), h], 1)
        h = h + self.pos(x.shape[-1])
        for block in self.blocks:
            h = block(h)
        return self.norm(h)[:, 0]


class DINOHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(DIM, HEAD_HIDDEN), nn.GELU(),
            nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN), nn.GELU(),
            nn.Linear(HEAD_HIDDEN, BOTTLENECK))
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        self.last = nn.utils.parametrizations.weight_norm(
            nn.Linear(BOTTLENECK, PROTOTYPES, bias=False))
        g = self.last.parametrizations.weight.original0
        g.data.fill_(1.0)
        g.requires_grad = False

    def forward(self, x):
        return self.last(F.normalize(self.mlp(x), dim=-1))


class DINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ViT()
        self.head = DINOHead()


def cosine(start, end, epochs, steps, warmup_epochs=0, warmup_from=0.0):
    warm = [warmup_from + (start - warmup_from) * i / (warmup_epochs * steps)
            for i in range(min(warmup_epochs, epochs) * steps)]
    rest = (epochs - warmup_epochs) * steps
    return warm + [end + 0.5 * (start - end) * (1 + math.cos(math.pi * i / rest))
                   for i in range(max(rest, 0))]


def teacher_temp(epoch):
    lo, hi = TEACHER_TEMP
    return lo + (hi - lo) * min(epoch, TEACHER_TEMP_EPOCHS) / TEACHER_TEMP_EPOCHS


def param_groups(model):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        flat = p.ndim <= 1 or name.endswith(".bias")
        embed = name.endswith("pos_embed") or name.endswith("cls_token")
        (no_decay if flat or embed else decay).append(p)
    return [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]


def dino_loss(student, teacher, center, temp):
    s = F.log_softmax(student / STUDENT_TEMP, -1).chunk(GLOBAL_CROPS + LOCAL_CROPS)
    t = F.softmax((teacher - center) / temp, -1).chunk(GLOBAL_CROPS)
    terms = [-(ti * sv).sum(-1).mean()
             for i, ti in enumerate(t) for v, sv in enumerate(s) if v != i]
    return sum(terms) / len(terms)


def stratified_split(labels):
    gen = torch.Generator().manual_seed(SPLIT_SEED)
    select = []
    for c in range(10):
        idx = (labels == c).nonzero().flatten()
        select.append(idx[torch.randperm(len(idx), generator=gen)[:SELECT_PER_CLASS]])
    select = torch.cat(select).sort().values
    mask = torch.ones(len(labels), dtype=torch.bool)
    mask[select] = False
    return mask.nonzero().flatten(), select


@torch.no_grad()
def features(backbone, images, mean, std, autocast):
    was = backbone.training
    backbone.eval()
    out = []
    for i in range(0, len(images), FEATURE_BATCH):
        x = (images[i:i + FEATURE_BATCH].float() / 255.0 - mean) / std
        with autocast():
            out.append(backbone(x).float())
    backbone.train(was)
    return torch.cat(out)


def standardise(fit, *others):
    mu, sd = fit.mean(0), fit.std(0).clamp_min(1e-6)
    return [(f - mu) / sd for f in (fit, *others)]


def fit_linear(x, y, wd):
    w = torch.zeros(x.shape[1], 10, device=x.device, requires_grad=True)
    b = torch.zeros(10, device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=PROBE_ITERS,
                            tolerance_grad=1e-7, tolerance_change=1e-10,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x @ w + b, y) + 0.5 * wd * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def accuracy(w, b, x, y):
    return ((x @ w + b).argmax(1) == y).float().mean().item()


@torch.no_grad()
def knn(fit_x, fit_y, x, y, leave_one_out=False):
    fit_n, q_n = F.normalize(fit_x, dim=1), F.normalize(x, dim=1)
    correct = 0
    for i in range(0, len(q_n), 1000):
        sim = q_n[i:i + 1000] @ fit_n.T
        if leave_one_out:
            rows = torch.arange(sim.shape[0], device=sim.device)
            sim[rows, rows + i] = -float("inf")
        top, idx = sim.topk(KNN_K, dim=1)
        votes = torch.zeros(sim.shape[0], 10, device=sim.device)
        votes.scatter_add_(1, fit_y[idx], torch.exp(top / KNN_T))
        correct += (votes.argmax(1) == y[i:i + 1000]).sum().item()
    return correct / len(y)


def probe_suite(fit, select, test, fit_y, select_y, test_y):
    fit, select, test = standardise(fit, select, test)
    sweep = {}
    for wd in PROBE_WDS:
        w, b = fit_linear(fit, fit_y, wd)
        sweep[wd] = (accuracy(w, b, select, select_y), w, b)
    best = max(PROBE_WDS, key=lambda wd: sweep[wd][0])
    _, w, b = sweep[best]
    return {
        "linear": {"weight_decay": best,
                   "train": accuracy(w, b, fit, fit_y),
                   "val": sweep[best][0],
                   "test": accuracy(w, b, test, test_y),
                   "sweep": {str(wd): sweep[wd][0] for wd in PROBE_WDS}},
        "knn": {"train": knn(fit, fit_y, fit, fit_y, leave_one_out=True),
                "val": knn(fit, fit_y, select, select_y),
                "test": knn(fit, fit_y, test, test_y)},
    }


def ema_(teacher, student, m):
    with torch.no_grad():
        t = list(teacher.parameters())
        torch._foreach_mul_(t, m)
        torch._foreach_add_(t, list(student.parameters()), alpha=1 - m)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/dino")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data", default=os.environ.get(
        "RKTORCH_CIFAR10_DIR", os.path.expanduser("~/.racket/rktorch/cifar10")))
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    def autocast():
        return torch.autocast(device.type, dtype=torch.bfloat16,
                              enabled=args.precision == "bf16")

    (train_x, train_y), (test_x, test_y) = load_cifar10(args.data)
    fit_idx, select_idx = stratified_split(train_y)
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)
    fit_idx, select_idx = fit_idx.to(device), select_idx.to(device)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    torch.manual_seed(args.seed)
    student = DINO().to(device)
    teacher = copy.deepcopy(student).requires_grad_(False)
    random_init = copy.deepcopy(student.backbone).requires_grad_(False)
    gen = torch.Generator(device).manual_seed(args.seed + 1)

    steps = len(train_x) // BATCH
    lr_s = cosine(BASE_LR * BATCH / 256, MIN_LR, args.epochs, steps, WARMUP_EPOCHS)
    wd_s = cosine(WD[0], WD[1], args.epochs, steps)
    mom_s = cosine(MOMENTUM[0], MOMENTUM[1], args.epochs, steps)
    opt = torch.optim.AdamW(param_groups(student), lr=lr_s[0], weight_decay=wd_s[0])
    last_layer = [p for p in student.head.last.parameters() if p.requires_grad]
    center = torch.zeros(1, PROTOTYPES, device=device)

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({k: v for k, v in globals().items()
                   if k.isupper() and isinstance(v, (int, float, tuple, str))}
                  | vars(args) | {"torch": torch.__version__,
                                  "params": sum(p.numel() for p in student.parameters())},
                  f, indent=1)
    log = open(os.path.join(args.out, "metrics.jsonl"), "w")

    def knn_select(backbone):
        fs = features(backbone, train_x, mean, std, autocast)
        fit, select = standardise(fs[fit_idx], fs[select_idx])
        return knn(fit, train_y[fit_idx], select, train_y[select_idx])

    print(json.dumps({"epoch": 0, "knn_select": knn_select(teacher.backbone)}), flush=True)
    step = 0
    for epoch in range(args.epochs):
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        temp = teacher_temp(epoch)
        totals = torch.zeros(4, device=device)
        order = torch.randperm(len(train_x), device=device, generator=gen)
        for i in range(steps):
            for g in opt.param_groups:
                g["lr"] = lr_s[step]
            opt.param_groups[0]["weight_decay"] = wd_s[step]
            g_views, l_views = multi_crop(train_x[order[i * BATCH:(i + 1) * BATCH]],
                                          gen, mean, std)
            with autocast():
                with torch.no_grad():
                    t_feat = teacher.backbone(g_views)
                    t_out = teacher.head(t_feat).float()
                s_out = torch.cat([student.head(student.backbone(g_views)),
                                   student.head(student.backbone(l_views))]).float()
            loss = dino_loss(s_out, t_out, center, temp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if epoch < FREEZE_LAST_EPOCHS:
                for p in last_layer:
                    p.grad = None
            nn.utils.clip_grad_norm_(student.parameters(), CLIP)
            opt.step()
            ema_(teacher, student, mom_s[step])
            with torch.no_grad():
                center.mul_(CENTER_MOMENTUM).add_(t_out.mean(0, keepdim=True),
                                                  alpha=1 - CENTER_MOMENTUM)
                probs = F.softmax((t_out - center) / temp, -1)
                mean_p = probs.mean(0)
                totals += torch.stack([
                    loss.detach(),
                    -(probs * torch.log(probs + 1e-12)).sum(-1).mean(),
                    -(mean_p * torch.log(mean_p + 1e-12)).sum(),
                    F.normalize(t_feat.float(), dim=-1).std(0).mean()])
            step += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        loss_m, ent, mean_ent, feat_std = (totals / steps).tolist()
        row = {"epoch": epoch + 1, "loss": loss_m, "teacher_entropy": ent,
               "mean_entropy": mean_ent, "feature_std": feat_std,
               "lr": lr_s[step - 1], "wd": wd_s[step - 1],
               "momentum": mom_s[step - 1], "teacher_temp": temp,
               "seconds": seconds, "images_per_s": steps * BATCH / seconds,
               "peak_gb": (torch.cuda.max_memory_allocated() / 2**30
                           if device.type == "cuda" else 0.0)}
        if (epoch + 1) % KNN_EVERY == 0 or epoch + 1 == args.epochs:
            row["knn_select"] = knn_select(teacher.backbone)
            torch.save({"teacher": teacher.state_dict(), "student": student.state_dict(),
                        "center": center, "epoch": epoch + 1},
                       os.path.join(args.out, "last.pt"))
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()

    torch.save(teacher.backbone.state_dict(), os.path.join(args.out, "teacher.pt"))
    fit_y, select_y = train_y[fit_idx], train_y[select_idx]
    pixels = [x.float().flatten(1) / 255.0 for x in (train_x, test_x)]
    results = {}
    for name, (tr, te) in {
            "dino": [features(teacher.backbone, x, mean, std, autocast)
                     for x in (train_x, test_x)],
            "random_init": [features(random_init, x, mean, std, autocast)
                            for x in (train_x, test_x)],
            "pixels": pixels}.items():
        results[name] = probe_suite(tr[fit_idx], tr[select_idx], te,
                                    fit_y, select_y, test_y)
        print(json.dumps({name: results[name]}), flush=True)
    with open(os.path.join(args.out, "probe.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
