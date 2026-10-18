# Building from source

rktorch is developed with [Nix](https://nixos.org/). The flake builds the native
shim (`libtorchrkt`, a C++ library linked against libtorch), installs the Racket
package, and runs every test and example. Supported systems: `x86_64-linux` and
`aarch64-darwin`.

This page is the guide for people. [`AGENTS.md`](../AGENTS.md) carries the same
commands for coding agents, beside the repository's layout and conventions; a
change to a build target or a shell belongs in both.

## Build and test

```bash
nix build                 # native lib + Racket package + tests + examples
./result/bin/torch        # prints the libtorch version and a 2x2 draw
nix build .#cpp           # CMake build and the GoogleTest suite only
nix flake check           # the Nix checks: cpp, format, tidy, line gate, racket,
                          # racket91, racket-floor, matrix-runner-test,
                          # platform-audit-test (Linux) and the
                          # racket-deps-* checks
```

`racket-deps` (flake.nix) is the catalog packages `torch/info.rkt` declares
beyond the Racket distribution, fetched as source trees under one hash.
`racket-deps-current` fails with "info.rkt changed; bump racket-deps outputHash
in flake.nix" when the declared deps no longer match the fetched ones, and
names any dep whose `#:version` minimum the fetched tree or the Racket
installation does not meet;
`racket-deps-guard-test` and `racket-deps-info-test` test
`nix/racket-deps/racket-deps.rkt`, and the info test also fails when
`own-platform-packages` there and `platform-packages` in
`torch/foreign/raw/native-location.rkt` name different packages;
`racket-deps-provision-test` tests
`nix/racket-deps/provision.sh`, the dev shells' provisioning. To bump, set
`outputHash` to `""`, run `nix build .#racket-deps`, and paste the hash Nix
reports. torch's own platform packages, such as `torch-x86_64-linux`, are left
out of the fetch: the build and the shells install their library-less
skeleton from `pkgs/` instead (see [The platform package](#the-platform-package)),
after checking that the skeleton's version and the dep's `#:version` both
equal torch's. Any other `#:platform` dep is refused, whether `torch/info.rkt`
or a fetched package declares it: one hash cannot hold a per-system fetch. A
package source (a URL or path) is refused too: racket-deps fetches catalog
packages by name.

`nix develop` provisions `.racket-user` again whenever racket-deps, the Racket
version or `torch/info.rkt` changes. It runs the same drift check first and
stops on it, removes the packages only the previous racket-deps held, and
installs `torch` with `--deps fail`, so a dep missing from racket-deps fails
the shell entry instead of being fetched from the live catalog.

The package supports Racket 9.1 and later. `nix run .#matrix` builds and tests
every supported Racket against the pinned libtorch on the host's system.

CI runs two more jobs that `nix flake check` does not cover. The Resyntax gate
is the `scripts/resyntax analyze` command below, Resyntax's default rules plus
the repository's own in `scripts/resyntax-rules.rkt`, and it fails on any
suggestion. It lints the changes since the merge base with an optional `REF`
after the directory, which defaults to the open pull request's base from `gh`
(found by head commit on a detached HEAD) when origin has that branch, else
`origin/master`. Resyntax skips `#lang scribble/lp2`, so the wrapper lints the
literate examples' code chunks through `scripts/resyntax-lp2.rkt`. The
code-generation drift check regenerates the ATen bindings and fails if the
committed output differs:

```bash
nix run .#codegen && git status --porcelain   # must print nothing
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
scripts/resyntax analyze --local-git-repository .   # the lint gate
racket scripts/coverage.rkt       # expression coverage, with a floor
racket scripts/bench/run.rkt micro   # benchmarks, docs/benchmarks/README.md
```

The first entry into a shell installs the Racket dependencies into a
per-checkout `.racket-user` directory and stages `libtorchrkt` under
`torch/native-libs/`. Resyntax, racket-review and `cover` come pinned and
precompiled from `nix/racket-linters.nix`, which says how to bump them, so
that step needs no network. After changing C++, re-stage the library with
`nix run .#copy-native-libs` before running `raco test`.

## The native library

`torch` looks for `libtorchrkt` in two places, in order: the staged
`torch/native-libs/`, then the collection `torch-native`, which a platform
package (`torch-x86_64-linux`, `torch-aarch64-macosx`) provides. A staged
library wins, so a linked checkout runs the shim it built even with a platform
package installed. When neither has it, or what it found would not open, the
error says which. Nix stages it for you -- `nix build`, or the first entry into
`nix develop` -- and `nix run .#copy-native-libs` re-stages it after a C++
change. Without Nix there are two ways to supply it.

Set `TORCHRKT_NATIVE_LIB_PATH` to a directory whose `lib/` subdirectory holds
the library, and the package's pre-install hook copies it into place:

```bash
raco pkg install ./pkgs/torch-x86_64-linux     # x86_64 Linux: meets torch's platform dep
raco pkg install ./pkgs/torch-aarch64-macosx   # racket-lang.org's Racket on arm64 macOS
TORCHRKT_NATIVE_LIB_PATH=/path/to/prefix raco pkg install --name torch ./torch
# /path/to/prefix/lib/libtorchrkt.so   (libtorchrkt.dylib on darwin)
```

Or copy it in by hand, which the hook leaves alone:

```bash
cp libtorchrkt.so torch/native-libs/
```

When the variable is unset, the hook prints nothing if either place already
holds the library, as a catalog install's platform package does, and
otherwise prints the same not-found message the bindings raise.

A hand-built library has to find libtorch at load time as well; the one Nix
builds carries an rpath to it, so a copy from elsewhere may need
`LD_LIBRARY_PATH` (`DYLD_LIBRARY_PATH` on darwin) to point at libtorch's `lib/`.
A library that is staged but cannot resolve libtorch reports as found, with
the loader's own message.

The shim and the bindings must come from the same commit. `tr_abi_stamp()`
returns a SHA-256 of every header under `cpp/include/torchrkt/c_api/`, baked in
when CMake configures, and `torch` refuses a shim whose stamp differs from the
copy in `torch/foreign/raw/abi-stamp.rktd`, naming both stamps and the fix:
`raco pkg update torch torch-x86_64-linux` (the platform package for the host)
for a catalog install, or re-staging in a checkout. After editing a `c_api`
header, `nix run .#abi-stamp` rewrites the copy (`nix run .#codegen` does too),
and the `abi-stamp-current` flake check fails until it is committed.

## The platform package

`torch` depends on one platform package per supported host, each providing
the collection `torch-native` at torch's own version: `torch-x86_64-linux`
(`#:platform "x86_64-linux"`) and `torch-aarch64-macosx`
(`#:platform "aarch64-macosx"`). Each holds `libtorchrkt`, libtorch 2.14.0's
libraries from PyTorch's official zip, and libsndfile with its codecs as
separate shared libraries, and finds them all beside itself.

Their skeletons live in `pkgs/`: the `info.rkt`, a `THIRD-PARTY-NOTICES`,
and the licences only that platform bundles. The licences both packages carry
(PyTorch's, stb's and the codecs') live once, in
`scripts/platform/licenses/`, and `scripts/platform/assemble.sh` adds them
when it builds a package. The Nix build and the shells install the host's
skeleton, with no libraries in it, so `torch`'s dependency is met without a
download; the staged `torch/native-libs/` still supplies the shim.

`(system-library-subpath #f)` decides which dep applies. racket-lang.org's
Racket for macOS reports `aarch64-macosx`, but Nix's Racket on the same Mac
reports `aarch64-darwin`, so a Nix-built Racket on a Mac does not pull in
`torch-aarch64-macosx` and has to build the shim, as the Nix shells do.

Both builds share `scripts/platform/build-common.sh`, which fetches every
source, checks it against the sha256 that `scripts/platform/sources.txt` pins
(the libtorch zips, the codecs, libsndfile, stb, GoogleTest and the Racket
installers the macOS smoke uses), builds the codecs and libsndfile as shared
libraries, then GoogleTest and the shim, and runs the shim's gtests against
the finished bundle. The work directory keeps the downloads and builds, so a
second run rebuilds only the shim; the unpacked libtorch, the codecs and
GoogleTest are rebuilt when the sha256 pinned for them changes.

### x86_64 Linux

Every library carries the RUNPATH `$ORIGIN` and needs nothing from the host
beyond glibc 2.28 and libstdc++, which is what PyTorch's own libtorch asks
for. The package is about 118 MB zipped and 445 MB installed. It is built
outside Nix in PyTorch's own baseline image, `manylinux_2_28`, since a Nix
toolchain links against a newer glibc than that baseline.

```bash
scripts/platform/linux/build.sh .scratch/platform
scripts/platform/assemble.sh torch-x86_64-linux .scratch/platform/bundle .scratch/pkg
```

`build.sh` runs in Docker, capped at four CPUs. It collects the bundle by
following NEEDED entries from the shim through the libtorch zip and the codec
builds, audits it, and runs the gtests against it.

The audit, `scripts/platform/linux/audit-bundle.sh <dir>`, fails the build if
any library:

- mentions `/nix/store`;
- has an RPATH or RUNPATH entry other than `$ORIGIN`, or lacks `$ORIGIN`
  while it needs a library in the bundle;
- needs a GLIBC, GLIBCXX or CXXABI symbol version newer than libtorch's own
  (2.28, 3.4.22, 1.3.11);
- or needs a library that is neither in the bundle nor in the system set
  (libc, libm, libdl, libpthread, librt, the dynamic loader, libstdc++,
  libgcc_s).

It reads the symbol versions twice, from `objdump -T` and from `readelf -V`,
and fails when the two disagree, so a tool whose output it cannot parse fails
the audit instead of passing it. It follows NEEDED entries only, so a library
loaded with `dlopen` is neither bundled nor audited; the CPU bundle loads
none. The `platform-audit-test` flake check points it at the Nix-built shim,
which it must fail on `/nix/store`, RUNPATH and GLIBC, and at the same shim
with an `objdump` that prints nothing.

`assemble.sh` copies the skeleton and the shared licences, adds the bundle
and writes `torch-x86_64-linux.zip` with `raco pkg create`. To try an install
on a host without Nix, `catalog.sh` puts that zip and a zip of this
checkout's `torch` in a directory catalog, and `smoke.sh` installs both from
it into a distro image with a Racket from download.racket-lang.org. Inside
the image, `smoke-install.sh` installs into a Racket user home of its own,
`$SMOKE_HOME` or else a fresh temporary directory, whatever `PLTUSERHOME` it
inherits. It fails if `raco pkg install` wrote anything to stderr, then
trains `04-mlp` on the device it picks, and on the CPU as well when that is
an accelerator, printing `(mps-available?)` on the way, and round-trips a
FLAC file:

```bash
scripts/platform/catalog.sh .scratch/pkg/torch-x86_64-linux.zip .scratch/catalog
scripts/platform/linux/smoke.sh rockylinux:8 racket-9.1-x86_64-linux-buster-cs.sh .scratch/catalog
```

### arm64 macOS

The macOS package needs macOS 14.0 or later on Apple silicon, the `minos`
that libtorch 2.14's own dylibs carry; `scripts/platform/macos/deployment-target`
holds it, and the codecs, GoogleTest and the shim are all built for it. Every
library's id is `@rpath/<name>`, each one that loads a bundled library has the
single `LC_RPATH @loader_path`, and nothing else is loaded except libSystem,
libc++ and libobjc from `/usr/lib` and frameworks from `/System`. libtorch
uses the system's Accelerate and bundles no MKL; its OpenMP runtime,
`libomp.dylib`, ships in the package. After every `install_name_tool` edit the
library is re-signed ad hoc (`codesign --force --sign -`), since arm64 macOS
refuses to load code whose signature no longer matches.

A Linux host can neither build nor load any of this, so
`.github/workflows/platform-macos.yml` builds it, on `macos-latest`, for
every pull request that touches `cpp/`, `scripts/platform/`, `pkgs/` or
`torch/info.rkt` outside `scripts/platform/linux/` and
`pkgs/torch-x86_64-linux/`, and on demand. Each step is one call on a Mac:

```bash
scripts/platform/macos/build.sh .scratch/macos codecs gtest shim bundle audit gtests
scripts/platform/macos/audit-test.sh .scratch/macos/bundle .scratch/macos/libtorch/libtorch/lib
scripts/platform/macos/install-racket.sh 9.1 .scratch/racket-9.1 .scratch/racket-dl
PATH=$PWD/.scratch/racket-9.1/bin:$PATH scripts/platform/assemble.sh torch-aarch64-macosx \
  .scratch/macos/bundle .scratch/pkg
PATH=$PWD/.scratch/racket-9.1/bin:$PATH scripts/platform/catalog.sh \
  .scratch/pkg/torch-aarch64-macosx.zip "$PWD/.scratch/catalog" "$PWD/.scratch/catalog"
SMOKE_HOME=$PWD/.scratch/smoke-home PATH=$PWD/.scratch/racket-9.1/bin:$PATH \
  scripts/platform/smoke-install.sh "$PWD/.scratch/catalog" "macOS $(sw_vers -productVersion)"
```

`build.sh` needs bash 4 or later, CMake, `pkg-config` and Xcode's
command-line tools; Homebrew's `bash`, `cmake` and `pkgconf` serve, and no
library from Homebrew is used.

It collects the bundle by following load commands from the shim through the
libtorch zip and the codec builds. The audit,
`scripts/platform/macos/audit-bundle.sh <dir> <target>`, reads each dylib with
`otool -l` and fails the build if any:

- mentions `/nix/store`;
- has an id other than `@rpath/<name>`;
- loads a library outside `@rpath`, `@loader_path`, `/usr/lib` and
  `/System`, a `/usr/lib` library other than libSystem, libc++ and libobjc,
  or an `@rpath` or `@loader_path` library that is not in the bundle;
- has an `LC_RPATH` other than `@loader_path`, more than one, or none while
  it loads a bundled library through `@rpath`;
- has a `minos` above the target;
- or has a signature that `codesign --verify` rejects.

`audit-test.sh` is its negative control: the unmodified libtorch zip must fail
on `libomp.dylib`'s `/opt/llvm-openmp` id and, against a 13.0 target, on
`minos`, and a copy of the shim with a rewritten id, a `/nix/store`
`LC_RPATH`, loads moved to Homebrew and to `/usr/lib`, and its signature
removed must fail on each.

The `minos` check reads only each header; the floor itself is enforced at
compile time, where `-Werror=unguarded-availability` and
`-Werror=unguarded-availability-new` make a call into the SDK newer than the
target an error in the codecs and the shim.

The workflow then installs racket-lang.org's notarized Racket 9.1 and 9.3
from their `.dmg` and runs `smoke-install.sh` with each. It uploads the zip,
its `CHECKSUM`, the audit and every log as the run's `torch-aarch64-macosx`
artifact.

The arm64 runner exposes a GPU to its VM: the smoke prints
`mps-available? #t` there and trains `04-mlp` on `'mps` and on the CPU. What
it cannot show is a physical Apple-silicon Mac, so the one check left by hand
is the same smoke on one, with the artifact's zip in a directory catalog as
above. It must print `mps-available? #t` and pass. To try more against that
install, run Racket with `PLTUSERHOME` set to the `SMOKE_HOME` it used:

```bash
PLTUSERHOME=$PWD/.scratch/smoke-home PATH=$PWD/.scratch/racket-9.1/bin:$PATH \
  racket -l racket/base -l torch -e '(mps-available?)'
```

## The manual

The manual is one Scribble document, `torch/scribblings/torch.scrbl`, with
two parts beneath it: `guide.scrbl`, a narrative introduction whose
chapters live in `guide/`, and `reference.scrbl`, which includes the
per-area chapters. Keeping both halves in a single document is what makes
the links between them ordinary relative links.

Chapters share `torch/scribblings/common.rkt`, whose `torch-examples` form
binds every example to one evaluator, so the results in the manual are
produced by the library at build time. An example that would need a GPU or
a download must not be written as a live one.

The guide's "Worked examples" part weaves the literate programs under
`torch/scribblings/examples/` into the manual by collection path
(`guide/examples.scrbl` includes
`(submod torch/scribblings/examples/NN-name doc)`), so a
`raco pkg install --copy` of `torch/` renders them too. Their harnesses
(`examples/test/`) and PyTorch twins (`examples/python/`) stay outside the
package.

`raco setup` renders both into `torch/doc/` (gitignored), and `raco docs`
opens them:

```bash
nix develop .#ci --command raco setup --pkgs torch
nix develop .#ci --command raco docs
```

Links into Racket's own manuals still go through its `local-redirect`
mechanism, whose rewriting script is loaded over `file://`. That is right
for reading the docs locally and broken for serving them over HTTP, where
the script never runs and such a link lands on a "Redirections"
placeholder. `scripts/build-docs` renders a copy that can be served, with
those links sent to the web, and checks it for what `raco scribble` lets
pass: a binding documented twice, which it only warns about, and broken
links, the names a page renders in red. It lists each broken link as
`page -> name` and fails when their count differs from the one it records:
more means a change added one, fewer means the record should come down to
match, so fixed links cannot be spent on new ones:

```bash
nix develop .#ci --command scripts/build-docs     # renders into manual/
python3 -m http.server -d manual 8000             # then open /torch/
```

## Coverage

`scripts/coverage.rkt` instruments the library with
[`cover`](https://pkgs.racket-lang.org/package/cover), drives it with the test
suite, and prints expression coverage per area:

```bash
nix develop .#ci --command racket scripts/coverage.rkt
nix develop .#ci --command racket scripts/coverage.rkt --changed
```

It exits non-zero below the floor set in the script, so it works as a gate as
well as a report. `--changed` adds the files the branch touches and, for each,
the line numbers no test reaches. The HTML report lands in `coverage/`.

Cold it takes about a minute, less than a cold `raco test torch/`, because
`cover` compiles instrumented code in memory and never writes bytecode.
Accelerator-only branches cannot be covered on the wrong host: MPS code is
unreachable on Linux, and the CUDA arms need `nix develop .#cuda` on a machine
with a GPU.

## The libtorch source

`flake.nix` has a `torchSource` knob:

- `"bin"` (the default) uses the prebuilt `pkgs.libtorch-bin` derivation,
  pointed at PyTorch's libtorch 2.14.0 downloads (nixpkgs still ships 2.9.0):
  a small download and fast, cached CI. The CUDA build adds cuDNN 9.24, which
  the cu130 download no longer bundles. On Linux the shells' Python torch is
  PyTorch's own wheel of the same release, CPU or cu130, and the cu130 one
  loads the same CUDA libraries and cuDNN as the Racket side. On macOS it is
  still nixpkgs' torch. Parity is checked to a float tolerance either way.
- `"python"` builds against the same libtorch the Python `torch` package ships,
  so seeded draws are bit-exact against PyTorch, at the cost of a much heavier
  build. It fails at evaluation while that torch is older than 2.14.0, the
  version the vendored schema and the MPS kernels assume.

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
nix develop .#ocaml --command ocamlc -version
nix develop .#ocaml --command ocamlfind query torch
nix develop .#ocaml --command dune exec --root /path/to/ocaml-project ./main.exe
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
