import os
import sys

N = 256
MASK = (1 << 64) - 1


def mix(*keys):
    z = 0x9E3779B97F4A7C15
    for k in keys:
        z = (z + k * 0xBF58476D1CE4E5B9 + 0x94D049BB133111EB) & MASK
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        z ^= z >> 31
    return z


CIFAR_SIDE = 32
CIFAR_COLOURS = [(183, 72, 72), (72, 183, 72), (72, 72, 183),
                 (183, 183, 72), (183, 72, 183), (72, 183, 183),
                 (183, 128, 72), (128, 72, 183), (128, 128, 128),
                 (72, 128, 128)]
CIFAR_DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2),
                    (2, -1), (1, -2), (3, 1), (1, 3)]


def cifar_record(i):
    label = mix(1, i) % 10
    dx, dy = CIFAR_DIRECTIONS[label]
    period = 13 + mix(2, i) % 6
    phase = mix(3, i) % period
    planes = bytearray()
    for c in range(3):
        for y in range(CIFAR_SIDE):
            for x in range(CIFAR_SIDE):
                t = (dx * x + dy * y + phase) % period
                wave = 120 * abs(2 * t - period) // period - 60
                noise = mix(4, i, c, y, x) % 17 - 8
                planes.append(CIFAR_COLOURS[label][c] + wave + noise)
    return bytes([label]) + bytes(planes)


def cifar10(out):
    with open(os.path.join(out, "cifar10-256.bin"), "wb") as f:
        for i in range(N):
            f.write(cifar_record(i))


MNIST_SIDE, MNIST_CELL = 28, 3
MNIST_LEFT = (MNIST_SIDE - 5 * MNIST_CELL) // 2
MNIST_TOP = (MNIST_SIDE - 7 * MNIST_CELL) // 2
MNIST_GLYPHS = [
    ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    ["01110", "10001", "10001", "01111", "00001", "00010", "01100"]]


def mnist_record(i):
    label = (i + mix(5, i // 10)) % 10
    width = MNIST_CELL - mix(6, i) % 2
    slant = mix(7, i) % 2
    ink = 200 + mix(8, i) % 56
    pixels = bytearray(MNIST_SIDE * MNIST_SIDE)
    for r, row in enumerate(MNIST_GLYPHS[label]):
        left = MNIST_LEFT + slant * (6 - r) // 4
        top = MNIST_TOP + r * MNIST_CELL
        for c in (c for c, bit in enumerate(row) if bit == "1"):
            for y in range(top, top + MNIST_CELL):
                for x in range(left + c * MNIST_CELL,
                               left + c * MNIST_CELL + width):
                    pixels[y * MNIST_SIDE + x] = ink - mix(9, i, y, x) % 40
    return label, bytes(pixels)


def mnist(out):
    records = [mnist_record(i) for i in range(N)]
    with open(os.path.join(out, "mnist-256-images-idx3-ubyte"), "wb") as f:
        f.write(bytes([0, 0, 8, 3]))
        for dim in (N, MNIST_SIDE, MNIST_SIDE):
            f.write(dim.to_bytes(4, "big"))
        for _, pixels in records:
            f.write(pixels)
    with open(os.path.join(out, "mnist-256-labels-idx1-ubyte"), "wb") as f:
        f.write(bytes([0, 0, 8, 1]) + N.to_bytes(4, "big"))
        f.write(bytes(label for label, _ in records))


out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "torch", "vision", "fixtures")
cifar10(out)
mnist(out)
