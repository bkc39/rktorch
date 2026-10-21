"""Regenerates torch/vision/fixtures/cifar10-256.bin deterministically.

CIFAR-10 carries no licence, so the fixture is synthetic: 256 records in
the dataset's binary format, one label byte then the red, green and blue
planes of a 32 by 32 image. Each class is a colour and a stripe direction,
each record a phase and frequency of its class's stripes plus a little
noise, all drawn from a fixed integer hash:

    python3 scripts/gen-cifar10-fixture.py
"""
import math
import os

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "torch", "vision", "fixtures", "cifar10-256.bin")

N, SIDE, CLASSES = 256, 32, 10
MASK = (1 << 64) - 1


def mix(*keys):
    z = 0x9E3779B97F4A7C15
    for k in keys:
        z = (z + k * 0xBF58476D1CE4E5B9 + 0x94D049BB133111EB) & MASK
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        z ^= z >> 31
    return z


def unit(*keys):
    return mix(*keys) / float(MASK)


colours = [(200, 60, 60), (60, 200, 60), (60, 60, 200), (200, 200, 60),
           (200, 60, 200), (60, 200, 200), (230, 140, 40), (120, 80, 40),
           (150, 150, 150), (40, 40, 90)]


def record(i):
    label = mix(1, i) % CLASSES
    angle = math.pi * label / CLASSES
    freq = 0.25 + 0.25 * unit(2, i)
    phase = 2 * math.pi * unit(3, i)
    dx, dy = math.cos(angle) * freq, math.sin(angle) * freq
    planes = bytearray()
    for c in range(3):
        for y in range(SIDE):
            for x in range(SIDE):
                wave = math.sin(dx * x + dy * y + phase)
                noise = 24 * (unit(4, i, c, y, x) - 0.5)
                v = colours[label][c] + 70 * wave + noise
                planes.append(min(255, max(0, int(round(v)))))
    return bytes([label]) + bytes(planes)


with open(out, "wb") as f:
    for i in range(N):
        f.write(record(i))
