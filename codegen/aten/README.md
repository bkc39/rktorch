# Vendored ATen schema (pinned)

`native_functions.yaml` and `tags.yaml` are vendored verbatim from the
pytorch **v2.14.0** tag — the version of the C++ libtorch we link against —
NOT from the dev-shell python torch (2.12). The generator must see the
schema of the library it binds, so do not "refresh" these from a newer
torch; bump them only when the pinned libtorch itself is bumped.

Source:

- https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/aten/src/ATen/native/native_functions.yaml
- https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/aten/src/ATen/native/tags.yaml

sha256 at vendoring time (2026-09-29):

```
ae1a1589816320d622f1e77fd2ed8a1f826c214791858d2394a0e4ad3db0cdc7  native_functions.yaml
013f0f6de0b8503050999db8f68e82bc9f1b009d9955d54dd13e6a648ca2c5c0  tags.yaml
```

These files are parsed by the dev shell's `torchgen` (from python torch
2.12) via `torchgen.gen.parse_native_yaml`. Parsing a 2.14 schema with a
2.12 torchgen is verified by the codegen self-check; if a future torchgen
breaks on this schema, pin torchgen via a fixed-output derivation in
flake.nix (same pattern as `racket-deps`).
