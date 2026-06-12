# Vendored ATen schema (pinned)

`native_functions.yaml` and `tags.yaml` are vendored verbatim from the
pytorch **v2.9.0** tag — the version of the C++ libtorch we link against —
NOT from the dev-shell python torch (2.12). The generator must see the
schema of the library it binds, so do not "refresh" these from a newer
torch; bump them only when the pinned libtorch itself is bumped.

Source:

- https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/aten/src/ATen/native/native_functions.yaml
- https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/aten/src/ATen/native/tags.yaml

sha256 at vendoring time (2026-06-12):

```
a97a636b21eca2b2e534ad248948bd20cf5471c81718be41a9f3b4488bf01db2  native_functions.yaml
61ca46c81accee71ad5191f961fbf62abb8aa330371a66cce8cb5a3efcb8b07d  tags.yaml
```

These files are parsed by the dev shell's `torchgen` (from python torch
2.12) via `torchgen.gen.parse_native_yaml`. Parsing a 2.9 schema with a
2.12 torchgen is verified by the codegen self-check; if a future torchgen
breaks on this schema, pin torchgen via a fixed-output derivation in
flake.nix (same pattern as `racket-deps`).
