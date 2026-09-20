`WavAndFlac/AudioRoundTrip.SaveInfoLoad` is parameterized over `rt.wav`,
`rt.flac`, `rt.WAV` and `rt.FLAC`, and every case writes
`testing::TempDir() + name`. On macOS the filesystem is case-insensitive,
so `rt.flac` and `rt.FLAC` name one file, and ctest starts the two cases
within 40 ms of each other. Whichever runs second reads a file the first
is rewriting or has already removed:

```
Start 134: WavAndFlac/AudioRoundTrip.SaveInfoLoad/"rt.flac"
Start 136: WavAndFlac/AudioRoundTrip.SaveInfoLoad/"rt.FLAC"
audio_test.cpp:55: Failure
tr_audio_info: cannot open /nix/var/nix/builds/.../rt.FLAC: Format not recognised.
```

That is a flake, not a regression — it hit
[#158's macOS job](https://github.com/bkc39/rktorch/actions/runs/35457378620/job/105934957884)
today and will keep hitting any PR whose macOS run happens to interleave
the two. Distinct stems (`rt1.wav`, `rt2.flac`, `rt3.WAV`, `rt4.FLAC`)
keep the coverage of both extension cases and remove the shared name.

`nix build .#cpp` passes locally; the macOS job here is the real check.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01VDEpCNkMi2rmxjgRnCHhkp
