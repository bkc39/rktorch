"""PyTorch twin of scripts/bench/workloads/workers.rkt, the e2e form of
torch/scribblings/examples/17-workers.rkt: the MNIST convnet trained from a
DataLoader with WORKERS worker processes, batches of 64. VARIANT=pgm
stores each training image as plain-PGM text and decodes, rotates and
shifts it in Python as it is loaded (the chapter's pure-Racket decode,
written the same way in pure Python); VARIANT=crop pads each image by two
pixels and cuts a random 28x28 window with torch, as commit 2b7ef04's
random-crop dataset did. The loss is read every step."""

import math
import random
import struct

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common import (StepLog, env_number, env_text, load_mnist, pick_device,
                    steps_out)
from mnist import accuracy, convnet

SIDE = 28


def pgm_encode(pixels):
    rows = (" ".join(str(v) for v in pixels[r * SIDE:(r + 1) * SIDE])
            for r in range(SIDE))
    return ("P2\n28 28\n255\n" + "".join(row + "\n" for row in rows)).encode()


def pgm_decode(bs):
    return bytes(int(v) for v in bs.split()[4:])


def pixel_at(pixels, x, y):
    return pixels[x + SIDE * y] if -1 < x < SIDE and -1 < y < SIDE else 0


def bilinear(pixels, x, y):
    x0, y0 = math.floor(x), math.floor(y)
    fx, fy = x - x0, y - y0
    return ((1 - fx) * (1 - fy) * pixel_at(pixels, x0, y0)
            + fx * (1 - fy) * pixel_at(pixels, x0 + 1, y0)
            + (1 - fx) * fy * pixel_at(pixels, x0, y0 + 1)
            + fx * fy * pixel_at(pixels, x0 + 1, y0 + 1))


def jitter(pixels, angle, dx, dy):
    c, s = math.cos(angle), math.sin(angle)
    out = []
    for y in range(SIDE):
        for x in range(SIDE):
            u, v = x - 13.5 - dx, y - 13.5 - dy
            out.append(bilinear(pixels, 13.5 + c * u + s * v,
                                13.5 + c * v - s * u) / 255.0)
    return struct.pack(f"{SIDE * SIDE}f", *out)


def jitter_draws(rng):
    return ((rng.random() - 0.5) * math.pi / 6,
            4 * (rng.random() - 0.5),
            4 * (rng.random() - 0.5))


class EncodedDigits(Dataset):
    def __init__(self, images, labels):
        raw = (images * 255 + 0.5).to(torch.uint8).reshape(len(images), -1)
        self.pgms = [pgm_encode(bytes(row.tolist())) for row in raw]
        self.labels = labels.tolist()
        self.rng = random.Random(0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        pixels = pgm_decode(self.pgms[i])
        data = jitter(pixels, *jitter_draws(self.rng))
        image = torch.frombuffer(bytearray(data), dtype=torch.float32)
        return image.reshape(1, SIDE, SIDE), self.labels[i]


class Jittered(Dataset):
    def __init__(self, images, labels):
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        padded = F.pad(self.images[i].unsqueeze(0), (2, 2, 2, 2))
        top, left = torch.randint(0, 5, (2,)).tolist()
        return padded[0, :, top:top + SIDE, left:left + SIDE], self.labels[i]


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    info = torch.utils.data.get_worker_info()
    if hasattr(info.dataset, "rng"):
        info.dataset.rng = random.Random(seed)


def main():
    out = steps_out()
    epochs = env_number("EPOCHS", 2)
    workers = env_number("WORKERS", 0)
    batch = env_number("BATCH", 64)
    variant = env_text("VARIANT", "pgm")
    dataset = env_text("DATASET", "full")
    torch.backends.cudnn.benchmark = False
    device = pick_device()
    log = StepLog(out, device)
    cpu = torch.device("cpu")
    train_x, train_y = load_mnist("train", cpu, dataset)
    test_x, test_y = ((train_x, train_y) if dataset == "fixture"
                      else load_mnist("test", cpu, dataset))
    ds = (Jittered(train_x, train_y) if variant == "crop"
          else EncodedDigits(train_x, train_y))
    torch.manual_seed(0)
    with torch.device(device):
        net = convnet()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    xs, ys = test_x.to(device), test_y.to(device)
    loader = DataLoader(ds, batch_size=batch, shuffle=True,
                        num_workers=workers, worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(0))
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(net(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
            log.step()
            loss.item()
        print(f"test acc {accuracy(net, xs, ys)}", flush=True)
    log.finish()


if __name__ == "__main__":
    main()
