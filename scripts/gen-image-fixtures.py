"""Regenerates torch/vision/fixtures/images deterministically.

Every image is drawn from a formula, so the fixtures carry no licence and
the lossless ones can be checked pixel for pixel without a decoder:

    nix develop --command python3 scripts/gen-image-fixtures.py

The PNGs are the gradients below, exactly. The JPEGs encode a smooth
field through Pillow's libjpeg, so their bytes depend on its version but
any two decoders agree on them to within a count or two; the reader's
parity test measures by how much.
"""
import math
import os

from PIL import Image

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "torch", "vision", "fixtures", "images")
os.makedirs(out, exist_ok=True)

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
smooth_image(401, 299).save(os.path.join(out, "smooth-401x299.jpg"),
                            quality=90, subsampling=2)

for name in sorted(os.listdir(out)):
    print(f"wrote {os.path.join(out, name)}")
