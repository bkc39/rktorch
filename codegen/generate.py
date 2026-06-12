"""Orchestrate generation: parse schema, filter by allowlist, classify,
emit all layers, report. Run as `nix run .#codegen` (or `python3 -m
codegen` in the dev shell); output paths resolve relative to this file,
so cwd only matters for the module lookup."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from torchgen.gen import parse_native_yaml

from . import emit_cpp, emit_racket
from .ir import Op, Skip, classify

ROOT = Path(__file__).resolve().parents[1]
ATEN = ROOT / "codegen" / "aten"
ALLOWLIST = ROOT / "codegen" / "allowlist.txt"

CPP_INCLUDE = ROOT / "cpp" / "include" / "torchrkt" / "c_api" / "generated"
CPP_UMBRELLA = ROOT / "cpp" / "include" / "torchrkt" / "c_api" / "generated.h"
CPP_SRC = ROOT / "cpp" / "src" / "torchrkt" / "generated"
RKT_RAW = ROOT / "torch" / "foreign" / "generated"
RKT_WRAPPERS = ROOT / "torch" / "generated.rkt"
MANIFEST = ROOT / "torch" / "tests" / "generated-parity.rktd"


def read_allowlist() -> list[tuple[str, str]]:
    entries = []
    for raw_line in ALLOWLIST.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        shard, _, op = line.partition(" ")
        if not op:
            sys.exit(f"allowlist: malformed line (want '<shard> <op>'): "
                     f"{raw_line!r}")
        entries.append((shard, op.strip()))
    return entries


def _clean(directory: Path, suffixes: tuple[str, ...]) -> None:
    if not directory.is_dir():
        return
    for child in directory.iterdir():
        if child.is_file() and child.suffix in suffixes:
            child.unlink()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _clang_format(paths: list[Path]) -> None:
    exe = shutil.which("clang-format")
    if exe is None:
        print("codegen: clang-format not on PATH; skipping format pass",
              file=sys.stderr)
        return
    subprocess.run([exe, "-i", *(str(p) for p in paths)], check=True)


def main() -> None:
    parsed = parse_native_yaml(
        str(ATEN / "native_functions.yaml"), str(ATEN / "tags.yaml")
    )
    by_name = {str(f.func.name): f for f in parsed.native_functions}

    shards: dict[str, list[Op]] = {}
    skips: list[Skip] = []
    for shard, name in read_allowlist():
        f = by_name.get(name)
        if f is None:
            sys.exit(f"allowlist: {name!r} not found in native_functions.yaml")
        result = classify(f, shard)
        if isinstance(result, Skip):
            skips.append(result)
        else:
            shards.setdefault(shard, []).append(result)
    for ops in shards.values():
        ops.sort(key=lambda o: o.c_name)
    shard_names = sorted(shards)

    _clean(CPP_INCLUDE, (".h",))
    _clean(CPP_SRC, (".cpp", ".cmake"))
    _clean(RKT_RAW, (".rkt",))

    cpp_paths = []
    for shard in shard_names:
        header = CPP_INCLUDE / f"{shard}.h"
        source = CPP_SRC / f"{shard}.cpp"
        _write(header, emit_cpp.emit_header(shard, shards[shard]))
        _write(source, emit_cpp.emit_source(shard, shards[shard]))
        cpp_paths += [header, source]
        _write(
            RKT_RAW / f"{shard}.rkt",
            emit_racket.emit_raw_shard(shard, shards[shard]),
        )
    _write(CPP_UMBRELLA, emit_cpp.emit_umbrella(shard_names))
    cpp_paths.append(CPP_UMBRELLA)
    _write(CPP_SRC / "sources.cmake", emit_cpp.emit_sources_cmake(shard_names))
    _write(RKT_WRAPPERS, emit_racket.emit_wrappers(shards))
    _write(MANIFEST, emit_racket.emit_manifest(shards))
    _clang_format(cpp_paths)

    total = sum(len(ops) for ops in shards.values())
    print(f"codegen: generated {total} op(s) across "
          f"{len(shard_names)} shard(s): {', '.join(shard_names)}")
    for skip in skips:
        print(f"codegen: SKIPPED {skip.aten_name} ({skip.shard}): "
              f"{skip.reason}")
