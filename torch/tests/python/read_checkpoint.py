"""Reads a rktorch checkpoint from Python, for checkpoint-format-parity-test.rkt
(T3-L7 of the T3 spec, #386).

python read_checkpoint.py <file.ckpt> <out.json> writes the header's
__metadata__, the checkpoint tree's top-level keys, every {"$tensor": entry}
reference in the tree and every tensor entry with its dtype tag, shape and
bytes in hex. safetensors.safe_open reads the metadata and the entry names
when the package is installed; the bytes always come from the file's own
offsets, so no framework is needed for bfloat16 or bool.
"""

import json
import struct
import sys


def by_offsets(path):
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + n].decode("utf-8"))
    metadata = header.pop("__metadata__", {})
    body = data[8 + n:]
    entries = {name: {"dtype": e["dtype"], "shape": e["shape"],
                      "hex": body[e["data_offsets"][0]:e["data_offsets"][1]].hex()}
               for name, e in header.items()}
    return metadata, entries


def by_safe_open(path):
    try:
        from safetensors import safe_open
    except ImportError:
        return None
    with safe_open(path, framework="numpy") as f:
        return f.metadata(), sorted(f.keys())


def references(tree, out):
    if isinstance(tree, dict):
        if set(tree) == {"$tensor"}:
            out.append(tree["$tensor"])
        for v in tree.values():
            references(v, out)
    elif isinstance(tree, list):
        for v in tree:
            references(v, out)
    return out


def main():
    path, out = sys.argv[1], sys.argv[2]
    metadata, entries = by_offsets(path)
    opened = by_safe_open(path)
    if opened is not None:
        assert opened[0] == metadata, "safe_open reads other metadata"
        assert opened[1] == sorted(entries), "safe_open reads other entries"
    tree = json.loads(metadata["checkpoint"])
    with open(out, "w") as f:
        json.dump({"reader": "offsets" if opened is None else "safe_open",
                   "metadata": {k: v for k, v in metadata.items() if k != "checkpoint"},
                   "keys": sorted(tree), "references": references(tree, []),
                   "entries": entries}, f, indent=1)


if __name__ == "__main__":
    main()
