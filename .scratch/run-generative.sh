#!/usr/bin/env bash
# the DCGAN and VAE headline runs on the GPU, grids under ~/mnist-generative
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
export OUT="$HOME/mnist-generative"
mkdir -p "$OUT"
export PLTCOLLECTS="/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152:"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export RKTORCH_MNIST_DIR="$HOME/.racket/rktorch/mnist"
noise="tree '/home|^Staging|^raco setup|^Done. Lint|full sweep|CUDA shell ready|raco test torch"
echo "dcgan start $(date -u +%H:%M)" | tee "$OUT/train.log"
EPOCHS=5 nix develop .#cuda -c racket -l racket/base -e '(require (submod "examples/test/10-dcgan.rkt" main))' 2>&1 | grep -vE "$noise" | tee -a "$OUT/train.log"
echo "vae start $(date -u +%H:%M)" | tee -a "$OUT/train.log"
EPOCHS=10 nix develop .#cuda -c racket -l racket/base -e '(require (submod "examples/test/11-vae.rkt" main))' 2>&1 | grep -vE "$noise" | tee -a "$OUT/train.log"
echo "generative done $(date -u +%H:%M)" | tee -a "$OUT/train.log"
ls "$OUT"
