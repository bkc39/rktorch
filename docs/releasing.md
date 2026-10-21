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

Racket versions take no suffix, so a pre-release lives in the tag:
`v0.0.1-alpha` releases `version` 0.0.1. The workflow's `plan` job fails
unless:

- the tag is `v<version>` or `v<version>-<suffix>`;
- `flake.nix`'s `version` equals `torch/info.rkt`'s;
- each platform dep's `#:version` and each `pkgs/*/info.rkt` `version` equal
  `native-version`;
- `scripts/release/release.rktd` lists exactly the packages that
  `torch/info.rkt` declares with `#:platform`, and each LGPL source it names
  is pinned in `scripts/platform/sources.txt`;
- the ABI stamp equals the one the last published natives were built with,
  or `native-version` has moved past theirs.

The last published natives are read from the newest published release that
carries a `manifest.json`. When `native-version` and the ABI stamp both match
it, the release builds no platform package: its manifest and catalog entries
point at the earlier release's assets. A shim change that leaves the headers
alone keeps the stamp, so bump `native-version` for it by hand, or it ships
in no platform package.

## Tagging

```bash
# torch/info.rkt and flake.nix carry the new version; native-version too, if the natives changed
git tag -a v0.0.1-alpha -m "torch 0.0.1-alpha"
git push origin v0.0.1-alpha
```

The run calls `platform-linux.yml` and `platform-macos.yml`, the same build,
audit and install jobs CI runs, when the natives build. Its `draft` job then
creates the draft release, a pre-release when the tag has a suffix. A draft
for the tag that exists already fails the run: delete the draft and re-run.

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
  the tag that built the natives, and each zip's and tarball's size and
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

- `pkg-build.yml`, dispatched with `platform_release: <tag>`, runs the
  package build server against a catalog whose `torch-x86_64-linux` is the
  published zip's https URL, before the next server build does the same.
- The package server's build, docs and test columns for `torch` are green,
  and the manual renders with its example chapters.
- `release-canary.yml`, dispatched: `raco pkg install torch` from
  pkgs.racket-lang.org on `ubuntu-latest` and `macos-14`, with Racket 9.1
  and stable, trains `04-mlp` and checks the package deps. It also runs
  weekly, to catch catalog drift and new Racket releases; until `torch` is on
  the catalog the weekly run passes with a notice, and a dispatched run fails.
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
2. A line in `scripts/release/release.rktd`: the package's name and its zip's
   path under the directory the `draft` job downloads every `torch-*`
   artifact into.
3. A job in `release.yml` calling the workflow that builds it, added to the
   `draft` job's `needs` and its `if`.
