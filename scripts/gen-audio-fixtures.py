"""Regenerates torch/audio/fixtures deterministically.

The WAV comes from python's stdlib wave module and the FLAC from the
reference encoder, so both stay independent of the code under test:

    python3 scripts/gen-audio-fixtures.py
    nix run nixpkgs#flac -- --silent --force \
        -o torch/audio/fixtures/sine-440-16k.flac \
        torch/audio/fixtures/sine-440-16k.wav
"""
import math
import os
import struct
import wave

path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "torch", "audio", "fixtures", "sine-440-16k.wav")
with wave.open(path, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes(b"".join(
        struct.pack("<h", round(16383 * math.sin(2 * math.pi * 440 * k
                                                 / 16000)))
        for k in range(1600)))
print(f"wrote {path}")
