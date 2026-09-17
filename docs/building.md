# Building from source

rktorch is developed with [Nix](https://nixos.org/). The flake builds the native
shim (`libtorchrkt`, a C++ library linked against libtorch), installs the Racket
package, and runs every test and example. Supported systems: `x86_64-linux` and
`aarch64-darwin`.

## Build and test

```bash
nix build                 # native lib + Racket package + tests + examples
./result/bin/torch        # prints the libtorch version and a 2x2 draw
nix build .#cpp           # CMake build and the GoogleTest suite only
nix flake check           # everything CI runs: cpp, format, tidy, line gate, racket
```

## Development shells

```bash
nix develop               # Racket, CMake, clang-tools, and a Python with torch
nix develop .#ci          # the same without Python torch (what the lint job uses)
nix develop .#cuda        # Linux: CUDA-linked shim and a CUDA Python torch
nix develop .#ocaml       # adds OCaml and Jane Street's Torch bindings
```

Inside a shell:

```bash
raco test torch/                  # unit tests; the PyTorch parity tests self-skip
                                  # where python3 cannot import torch
raco test examples/test/          # the literate examples
racket -ie "(require torch)"      # a REPL with the package loaded
resyntax analyze --local-git-repository . origin/master   # the lint gate
```

The first entry into a shell installs the Racket dependencies into a
per-checkout `.racket-user` directory and stages `libtorchrkt` under
`torch/native-libs/`. After changing C++, re-stage the library with
`nix run .#copy-native-libs` before running `raco test`.

## The libtorch source

`flake.nix` has a `torchSource` knob:

- `"bin"` (the default) uses the prebuilt `pkgs.libtorch-bin`: a small download
  and fast, cached CI. Parity with Python torch is checked to a float tolerance,
  because the C++ and Python builds may differ in patch version.
- `"python"` builds against the same libtorch the Python `torch` package ships,
  so seeded draws are bit-exact against PyTorch, at the cost of a much heavier
  build.

## Accelerators

`#:device 'cuda` works on Linux and `#:device 'mps` on Apple Silicon;
`(accelerator-if-available)` picks whichever is present. MPS works from the
default shell on darwin. CUDA needs `nix develop .#cuda`, which links the shim
against the CUDA libtorch and stages the host's NVIDIA driver libraries.

## The OCaml reference shell

`nix develop .#ocaml` adds OCaml, Dune, Findlib, utop, ocamllsp, ocamlformat,
and Jane Street's [Torch bindings](https://github.com/janestreet/torch), which
rktorch uses as a reference design:

```bash
nix develop .#ocaml --command ocamlfind query torch
nix develop .#ocaml --command utop
```

Dune projects can use `(libraries torch)` without an opam switch. The bindings
are pinned in [`nix/ocaml-torch.nix`](../nix/ocaml-torch.nix) to v0.17.0 with its
compatible libtorch 2.1.2; on Apple Silicon the native library comes from the
PyTorch wheel. The bindings' upstream inline tests are not run: some SVD
expectations are host-dependent up to sign, and the suite is pinned upstream
code rather than a regression surface for this repository. No other shell
includes the OCaml toolchain.

## Continuous integration and the binary cache

CI runs `nix flake check` on Linux and macOS, a Resyntax lint gate, and a
code-generation drift check. Linux jobs substitute from a tailnet-only binary
cache; the details and the failure modes are in [`AGENTS.md`](../AGENTS.md),
which is also the canonical guide to the repository's layout and conventions.

## Further reading

- [`docs/internals.md`](internals.md): how native memory is managed across the
  GC and FFI boundary
- [`docs/design/v1-codegen-nn.md`](design/v1-codegen-nn.md): the code generator
  and the design of the `nn` layer
