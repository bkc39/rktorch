"""Convert every P6 PPM in a directory to PNG beside it, with an index.html; no PIL needed."""
import os
import struct
import sys
import zlib


def read_ppm(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:2] == b"P6"
    parts = data.split(b"\n", 3)
    w, h = map(int, parts[1].split())
    return w, h, parts[3]


def write_png(path, w, h, rgb, scale=3):
    rows = []
    for y in range(h):
        row = rgb[y * w * 3:(y + 1) * w * 3]
        wide = b"".join(row[i:i + 3] * scale for i in range(0, len(row), 3))
        rows.append(b"\x00" + wide)
    raw = b"".join(r for r in rows for _ in range(scale))

    def chunk(tag, body):
        return (struct.pack(">I", len(body)) + tag + body
                + struct.pack(">I", zlib.crc32(tag + body) & 0xffffffff))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w * scale, h * scale, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


d = sys.argv[1]
names = sorted(n for n in os.listdir(d) if n.endswith(".ppm"))
for n in names:
    w, h, rgb = read_ppm(os.path.join(d, n))
    write_png(os.path.join(d, n[:-4] + ".png"), w, h, rgb)
with open(os.path.join(d, "index.html"), "w") as f:
    f.write("<title>MNIST DCGAN and VAE grids</title><body style='background:#222;color:#ddd;font-family:sans-serif'>")
    for n in names:
        f.write(f"<h3>{n[:-4]}</h3><img src='{n[:-4]}.png' style='image-rendering:pixelated'>\n")
    f.write("</body>")
print("converted", len(names))
