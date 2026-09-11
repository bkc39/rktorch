# torch

Racket bindings to **libtorch** (the C++ core of PyTorch).

> **Status: v0 scaffold.** A thin vertical slice — seed the RNG, draw a tensor,
> read it back, verify against PyTorch — wired through a full
> Nix + CMake + `raco test` + Resyntax + CI pipeline. The API is intentionally
> tiny; see [`plans/v0-scaffold.md`](plans/v0-scaffold.md) for scope and roadmap,
> and [`AGENTS.md`](AGENTS.md) for the build/dev guide.

```racket
(require torch)

(torch-version)            ; => "2.9.0"
(manual-seed! 0)
(define t (randn 2 2))     ; => #<tensor:2x2>
(tensor-shape t)           ; => '(2 2)
(tensor->list t)           ; => '(...four floats...)
(display (tensor->string t))
```

## Quick start

```bash
nix build              # build native lib, install pkg, run tests
./result/bin/torch  # prints the libtorch version and a 2x2 draw
nix develop            # dev shell (raco test, cmake, resyntax, ...)
```

The optional `ocaml` shell adds OCaml, Dune, Findlib, utop, ocamllsp,
ocamlformat, and Jane Street's Torch bindings for reference experiments:

```bash
nix develop .#ocaml --command ocamlc -version
nix develop .#ocaml --command ocamlfind query torch
nix develop .#ocaml --command dune exec --root /path/to/ocaml-project ./main.exe
nix develop .#ocaml --command utop
```

Dune projects can use `(libraries torch)` without an opam switch. The bindings
are pinned in [`nix/ocaml-torch.nix`](nix/ocaml-torch.nix) to Jane Street v0.17.0
and its compatible libtorch 2.1.2; on Apple Silicon the native library comes
from the PyTorch wheel. That runtime is linked separately into OCaml programs.
The first `ocaml` shell entry builds the bindings. Their upstream inline expect
tests are disabled because valid SVD sign differences make them host-dependent.
The default, `ci`, and `cuda` shells do not include the OCaml toolchain.

Built against `pkgs.libtorch-bin` by default; flip `torchSource` in `flake.nix`
to `"python"` for bit-exact PyTorch parity. Supported systems: `aarch64-darwin`,
`x86_64-linux`.

GPU: `#:device 'cuda` on Linux and `#:device 'mps` on Apple Silicon, or
`(accelerator-if-available)` to take whichever is present. MPS works out of the
default `nix develop` on darwin; CUDA needs `nix develop .#cuda`.

How native memory is managed across the GC/FFI boundary:
[docs/internals.md](docs/internals.md).

## License

Apache-2.0.
