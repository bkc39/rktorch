# exit-loop repro harness

These scripts were written to debug #72 (in-place native-lib staging corrupts
live processes). Open follow-up: #76. Kept so the failure can be re-run.

```bash
nix run .#copy-native-libs                                     # stage a matching shim
nix develop        --command ./scripts/debug/exit-loop/run-all.sh cpu  20
nix develop .#cuda --command ./scripts/debug/exit-loop/run-all.sh cuda 20
```

Reports land in `repro-logs/`. Knobs: `REPRO_DEVICE`, `REPRO_TIMEOUT`,
`REPRO_LOGDIR`. `corrupt-restage.sh` is destructive by design; it restores from
an EXIT trap. Each script's header says what it does.
