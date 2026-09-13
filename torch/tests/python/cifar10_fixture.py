"""The committed CIFAR-10 fixture parsed the way the Racket loader parses it.

Each record is one label byte and 3072 pixel bytes, channel-major; a pixel
maps to [-1, 1] by x / 127.5 - 1. The Racket twin compares labels, the
first pixels of the first image, and per-image means.
"""
import json
import os

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "..", "vision",
                       "fixtures", "cifar10-256.bin")
RECORD = 3073

with open(FIXTURE, "rb") as f:
    data = f.read()

n = len(data) // RECORD
labels = [data[i * RECORD] for i in range(n)]


def image(i):
    start = i * RECORD + 1
    return [b / 127.5 - 1.0 for b in data[start:start + RECORD - 1]]


first = image(0)
means = [sum(image(i)) / (RECORD - 1) for i in range(n)]

print(json.dumps({
    "n": n,
    "labels": labels,
    "first_pixels": first[:8] + first[1024:1028] + first[2048:2052],
    "means": means,
}))
