#!/usr/bin/env bash
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
nix develop .#cuda -c bash -c '
export LD_LIBRARY_PATH="$RKTORCH_CUDA_DRIVER_PATH"
python3 -c "
import torch
for v, dt in [(0.5, torch.int64), (2.0, torch.int64), (0.5, torch.uint8)]:
    try:
        print(v, dt, \"->\", torch.full((2,), v, dtype=dt).tolist())
    except Exception as e:
        print(v, dt, \"-> RAISED:\", str(e).split(chr(10))[0])
"'
