"""Regenerates torch/vision/fixtures/images deterministically.

Every image is drawn from a formula: the fixtures are the repository's
own work, under its Apache-2.0 license, and the lossless ones can be
checked pixel for pixel without a decoder:

    nix develop --command python3 scripts/gen-image-fixtures.py

The PNGs are the gradients below, exactly. The JPEGs encode a smooth
field through Pillow's libjpeg, so their bytes depend on its version; the
reader's parity test holds its decode of them to torchvision's, byte for
byte. smooth-cmyk.jpg is the same field as Adobe CMYK, with a varying K;
smooth-arithmetic.jpg is smooth.jpg's coefficients arithmetic-coded by
jpegtran. malformed-images/ holds two JPEGs that must not decode: one
with a 13-bit sample precision, one whose header claims 16000 x 16000.

The shim's gtests get copies of some of these in cpp/tests/torchrkt/data,
with image-goldens.txt: the shape and SHA-256 of torchvision's decode of
each in a mode, as "name mode C H W sha256" lines.
"""
import hashlib
import math
import os
import shutil
import subprocess

from PIL import Image
from torchvision.io import ImageReadMode, decode_image, read_file

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
fixtures = os.path.join(root, "torch", "vision", "fixtures")
out = os.path.join(fixtures, "images")
malformed = os.path.join(fixtures, "malformed-images")
gtest_data = os.path.join(root, "cpp", "tests", "torchrkt", "data")
for d in (out, malformed, gtest_data):
    os.makedirs(d, exist_ok=True)

W, H = 37, 23


def gradient(x, y):
    return (7 * x % 256, 11 * y % 256, (3 * x + 5 * y) % 256)


rgb = Image.new("RGB", (W, H))
rgb.putdata([gradient(x, y) for y in range(H) for x in range(W)])
rgb.save(os.path.join(out, "gradient.png"), optimize=True)

rgba = Image.new("RGBA", (W, H))
rgba.putdata([gradient(x, y) + (x * y % 256,)
              for y in range(H) for x in range(W)])
rgba.save(os.path.join(out, "gradient-rgba.png"), optimize=True)

gray = Image.new("L", (W, H))
gray.putdata([(5 * x + 3 * y) % 256 for y in range(H) for x in range(W)])
gray.save(os.path.join(out, "gradient-gray.png"), optimize=True)

palette = Image.new("P", (W, H))
palette.putpalette([200, 30, 30, 30, 200, 30, 30, 30, 200, 240, 240, 240])
palette.putdata([(x + y) % 4 for y in range(H) for x in range(W)])
palette.save(os.path.join(out, "palette.png"), optimize=True)


def smooth(x, y):
    return tuple(
        max(0, min(255, round(v)))
        for v in (128 + 90 * math.sin(x / 23) * math.cos(y / 31),
                  128 + 90 * math.cos((x + y) / 41),
                  128 + 60 * math.sin(x * y / 5000)))


def smooth_image(w, h):
    img = Image.new("RGB", (w, h))
    img.putdata([smooth(x, y) for y in range(h) for x in range(w)])
    return img


small = smooth_image(96, 64)
small.save(os.path.join(out, "smooth.jpg"), quality=90, subsampling=2)
small.save(os.path.join(out, "smooth-444.jpg"), quality=90, subsampling=0)
small.save(os.path.join(out, "smooth-progressive.jpg"), quality=90,
           subsampling=2, progressive=True)
small.convert("L").save(os.path.join(out, "smooth-gray.jpg"), quality=90)


def smooth_cmyk(x, y):
    r, g, b = smooth(x, y)
    return (255 - r, 255 - g, 255 - b, round(60 + 50 * math.sin(x / 17)))


cmyk = Image.new("CMYK", small.size)
cmyk.putdata([smooth_cmyk(x, y) for y in range(small.height)
              for x in range(small.width)])
cmyk.save(os.path.join(out, "smooth-cmyk.jpg"), quality=90)
smooth_image(401, 299).save(os.path.join(out, "smooth-401x299.jpg"),
                            quality=90, subsampling=2)

subprocess.run(["jpegtran", "-arithmetic",
                "-outfile", os.path.join(out, "smooth-arithmetic.jpg"),
                os.path.join(out, "smooth.jpg")], check=True)


def frame_at(data):
    at = 2
    while data[at + 1] != 0xC0:
        at += 2 + int.from_bytes(data[at + 2:at + 4], "big")
    return at


def with_frame(name, edit):
    with open(os.path.join(out, "smooth-gray.jpg"), "rb") as f:
        data = bytearray(f.read())
    edit(data, frame_at(data))
    for d in (malformed, gtest_data):
        with open(os.path.join(d, name), "wb") as f:
            f.write(data)


def precision_13(data, frame):
    data[frame + 4] = 13


def claims_16000(data, frame):
    data[frame + 5:frame + 9] = (16000).to_bytes(2, "big") * 2


with_frame("bad-precision.jpg", precision_13)
with_frame("huge.jpg", claims_16000)

goldens = [("smooth.jpg", "unchanged"), ("smooth.jpg", "gray"),
           ("smooth-cmyk.jpg", "unchanged"), ("smooth-cmyk.jpg", "rgb"),
           ("smooth-cmyk.jpg", "gray"), ("smooth-arithmetic.jpg", "unchanged")]
with open(os.path.join(gtest_data, "image-goldens.txt"), "w") as f:
    for name, mode in goldens:
        path = os.path.join(out, name)
        shutil.copyfile(path, os.path.join(gtest_data, name))
        pixels = decode_image(read_file(path), ImageReadMode[mode.upper()])
        digest = hashlib.sha256(pixels.contiguous().numpy().tobytes())
        f.write(" ".join([name, mode, *map(str, pixels.shape),
                          digest.hexdigest()]) + "\n")

for d in (out, malformed, gtest_data):
    for name in sorted(os.listdir(d)):
        print(f"wrote {os.path.join(d, name)}")
