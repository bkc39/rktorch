"""Records torch/vision/fixtures/vit-keys.rktd: the tensor names and
shapes in Hugging Face ViT checkpoints, read from each model.safetensors
header with two HTTP range requests at a pinned revision, so no weights
are downloaded and no `transformers` is needed:

    python3 scripts/record-vit-keys.py [OUT]

OUT defaults to the fixture. Each entry is
(repository revision classes ((key (dim ...)) ...)), keys sorted, with
classes #f for a checkpoint without a classifier.
"""
import json
import os
import struct
import sys
import urllib.request

REVISIONS = [
    ("google/vit-base-patch16-224",
     "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3", 1000),
    ("google/vit-base-patch16-224-in21k",
     "b4569560a39a0f1af58e3ddaf17facf20ab919b0", None),
]


def fetch(url, start, end):
    request = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=60) as reply:
        return reply.read()


def header(repo, revision):
    url = f"https://huggingface.co/{repo}/resolve/{revision}/model.safetensors"
    (n,) = struct.unpack("<Q", fetch(url, 0, 7))
    entries = json.loads(fetch(url, 8, 8 + n - 1))
    entries.pop("__metadata__", None)
    return sorted((k, v["shape"]) for k, v in entries.items())


def entry(repo, revision, classes, keys):
    rows = "\n".join(f"   ({json.dumps(k)} ({' '.join(map(str, shape))}))"
                     for k, shape in keys)
    label = "#f" if classes is None else str(classes)
    return f"({json.dumps(repo)} {json.dumps(revision)} {label}\n  (\n{rows}))"


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "torch", "vision", "fixtures", "vit-keys.rktd")
    body = "\n".join(entry(repo, revision, classes, header(repo, revision))
                     for repo, revision, classes in REVISIONS)
    with open(out, "w") as f:
        f.write(f"({body})\n")


if __name__ == "__main__":
    main()
