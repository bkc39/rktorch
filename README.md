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

Built against `pkgs.libtorch-bin` by default; flip `torchSource` in `flake.nix`
to `"python"` for bit-exact PyTorch parity. Supported systems: `aarch64-darwin`,
`x86_64-linux`.

How native memory is managed across the GC/FFI boundary:
[docs/internals.md](docs/internals.md).

## License

Apache-2.0.
