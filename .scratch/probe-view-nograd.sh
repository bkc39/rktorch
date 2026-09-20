#!/usr/bin/env bash
# python torch in the cuda shell needs the driver farm on LD_LIBRARY_PATH,
# not Racket's libtorch; python-env.rkt does the same for its children
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
nix develop .#cuda -c bash -c '
export LD_LIBRARY_PATH="$RKTORCH_CUDA_DRIVER_PATH"
python3 -c "
import torch
x = torch.rand(3,3,2,2, requires_grad=True) * 1.0
print(\"x\", x.requires_grad)
with torch.no_grad():
    print(\"select under no_grad:\", x.select(0,0).requires_grad)
    print(\"squeeze under no_grad:\", x.narrow(0,0,1).squeeze(0).requires_grad)
from torchvision.utils import make_grid
" 2>&1 | grep -v Warning'
