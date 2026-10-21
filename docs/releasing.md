# Releasing

A release is a tag on master. The tag runs `.github/workflows/release.yml`,
which checks the versions, builds the platform packages when their bytes
changed, and leaves a **draft** GitHub release. Nothing publishes itself:
the owner publishes the draft and edits the catalog entries by hand.

## Versions

| What | Where | Changes when |
|---|---|---|
| `version` | `torch/info.rkt`, and `version` in `flake.nix` | the Racket bindings change, so every release |
| `native-version` | `torch/info.rkt`; the `#:version` of every `#:platform` dep in it; `version` in each `pkgs/*/info.rkt` | the compiled bytes change: the shim, the libtorch pin or the build flags |
| ABI stamp | `torch/foreign/raw/abi-stamp.rktd` (`nix run .#abi-stamp`) | a `cpp/include/torchrkt/c_api` header changes |
| natives' inputs | a sha256 of `git ls-tree -r HEAD` over `natives-inputs` in `scripts/release/release.rktd`, less its `natives-inputs-exclude` paths | any tracked file the platform packages are built from changes |

Racket versions take no suffix, so a pre-release lives in the tag:
`v0.0.1-alpha` releases `version` 0.0.1. The workflow's `plan` job fails
unless:

- the tagged commit is on master, and the latest run of each status check
  master's rules require succeeded on it;
- the tag is `v<version>` or `v<version>-<suffix>`;
- `flake.nix`'s `version` equals `torch/info.rkt`'s;
- each platform dep's `#:version` and each `pkgs/*/info.rkt` `version` equal
  `native-version`;
- `scripts/release/release.rktd` has one zip for each package that
  `torch/info.rkt` declares with `#:platform`, and no other, and each LGPL
  source it names is pinned in `scripts/platform/sources.txt`;
- `version` is no older than the previous release's, and equal to it only
  under a different tag (`v0.0.1-alpha`, then `v0.0.1`);
- the ABI stamp and the natives' inputs equal those the previous release's
  natives were built with, or `native-version` has moved past theirs.

The previous release is the published release with a `manifest.json` whose
`native-version` is highest, the latest published among equals. When
`native-version`, the ABI stamp and the natives' inputs all match it, the
release builds no platform package: its manifest and catalog entries point at
the earlier release's assets. Any change to a file the natives are built from,
a shim source, a build script, a notice, needs a `native-version` bump, or
the run fails. A file that only tests or audits the build belongs in
`natives-inputs-exclude`.

## Tagging

`torch/info.rkt` and `flake.nix` carry the new `version`, and
`native-version` moves too when the natives changed:

```bash
git tag -a v0.0.1-alpha -m "torch 0.0.1-alpha"
git push origin v0.0.1-alpha
```

The run calls `platform-linux.yml` and `platform-macos.yml`, the same build,
audit and install jobs CI runs, when the natives build. When they don't, its
`carried` jobs install the earlier release's zips on Linux and macOS through
a catalog that names them by their https URLs, so a dead URL fails the run.
The `assemble` job, with read-only permissions, builds the assets; the
`draft` job, the only one that can write, attests them, creates the draft
release (a pre-release when the tag has a suffix) and fails unless the
draft's asset names and sizes are the assembled ones. A draft for the tag
that exists already fails the run: delete the draft and re-run.

A `workflow_dispatch` run is a dry run. It takes an optional `tag` to check
as if pushed, builds and assembles everything, uploads the assets as the
`release-assets` artifact, and drafts nothing.

## The draft's assets

- `<package>.zip` and `<package>.zip.CHECKSUM` for each platform package, when
  the natives build. The `.CHECKSUM` is the file `raco pkg` fetches beside a
  URL source and compares with the catalog's checksum.
- The source tarballs of the LGPL libraries the zips ship (libsndfile,
  mpg123, LAME), as pinned in `sources.txt`, when the natives build. LGPL-2.1
  §4 asks for the source beside the object code.
- `catalog-entry.rktd`: a hash from package name to its catalog entry, in
  the catalog's own format. Platform packages point at
  `https://github.com/bkc39/rktorch/releases/download/<tag>/<package>.zip`
  for the tag that built them; `torch` at
  `https://github.com/bkc39/rktorch.git?path=torch#<tag>`.
- `manifest.json`: the tag, commit, `version`, `native-version`, ABI stamp,
  natives' inputs, the tag that built the natives, and each zip's and tarball's size and
  sha256.
- `SHA256SUMS`, over every asset above.
- `provenance.sigstore.json`, the bundle of the build provenance attestation;
  `gh attestation verify <file> --repo bkc39/rktorch` checks any asset.

Every asset must be under 2 GiB, GitHub's limit, or the run fails.

## Publishing

In this order, so no user ever resolves a `torch` whose natives aren't
published:

1. Read the draft: the notes, `manifest.json`, the run's install jobs.
   Publish it.
2. When the release built the natives, edit each platform package's entry on
   pkgs.racket-lang.org to the source in `catalog-entry.rktd`. Otherwise
   leave them.
3. Edit `torch`'s entry to `https://github.com/bkc39/rktorch.git?path=torch#<tag>`.

## After publishing

- `pkg-build.yml`, dispatched with `release_tag: <tag>`, checks out the tag
  and runs the package build server against a catalog whose
  `torch-x86_64-linux` is the https URL of the zip under the tag's
  `manifest.json` `natives-tag`, before the next server build does the same.
- The package server's build, docs and test columns for `torch` are green,
  and the manual renders with its example chapters.
- `release-canary.yml`, dispatched: `raco pkg install torch` from
  pkgs.racket-lang.org on `ubuntu-latest` and `macos-14`, with Racket 9.1
  and stable, trains `04-mlp` and checks the package deps. It also runs
  weekly, to catch catalog drift and new Racket releases. Once a release is
  published, a catalog that doesn't have `torch` fails it.
- CUDA on the Nix route, on the lab host: `(cuda-available?)` and the MLP on
  `'cuda`.

## Rollback

Point the entries back: `torch`'s to the previous tag, and, if this release
built the natives, each platform package's to the previous natives' URL
(the previous release's `catalog-entry.rktd`). Never delete a published
release or asset: `raco pkg` re-downloads a package by its URL and checks its
checksum, so a deleted asset breaks every install that still resolves to it.
A draft can be deleted.

## Adding a platform package

1. Its skeleton under `pkgs/`, at `native-version`, and its `#:platform`
   deps in `torch/info.rkt` (and `platform-packages` in
   `torch/foreign/raw/native-location.rkt`, `own-platform-packages` in
   `nix/racket-deps/racket-deps.rkt`).
2. A line in `scripts/release/release.rktd`'s `platform-zips`: its zip's path
   under the directory the `assemble` job downloads every `torch-*` artifact
   into. The zip's file name is the package's name.
3. A job in `release.yml` calling the workflow that builds it, added to the
   `assemble` job's `needs` and its `if`, and a `carried` matrix entry for a
   runner that installs it.
