`torch` {{version}}, with platform packages {{native-version}}.

{{natives}}
`catalog-entry.rktd` holds the catalog entries this release needs. Publish
this release, then set any platform package's entry that changed, then
`torch`'s, which tracks the tag `{{tag}}`. `manifest.json` lists every
asset's size and sha256, and `SHA256SUMS` checks them. The build provenance
is attested by GitHub (`gh attestation verify <file> --repo bkc39/rktorch`).

The shim's ABI stamp is `{{abi-stamp}}`.

[`docs/releasing.md`](https://github.com/bkc39/rktorch/blob/{{tag}}/docs/releasing.md)
has the publishing order, the checks after publishing, and rollback.
