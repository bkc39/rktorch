#!/usr/bin/env bash
# the headline ResNet-18 run on the GPU: EPOCHS epochs, accuracy per epoch, into ~/cifar10-resnet/train.log
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
mkdir -p "$HOME/cifar10-resnet"
export PLTCOLLECTS="/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152:"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export RKTORCH_CIFAR10_DIR="$HOME/.racket/rktorch/cifar10"
export RKTORCH_MNIST_DIR="$HOME/.racket/rktorch/mnist"
export EPOCHS="${EPOCHS:-30}"
start=$(date +%s)
echo "start $(date -u +%H:%M) epochs $EPOCHS" | tee "$HOME/cifar10-resnet/train.log"
nix develop .#cuda -c racket -l racket/base -e '(require (submod "examples/test/09-resnet.rkt" main))' 2>&1 | grep -v "tree '/home\|^Staging\|^raco setup\|^Done. Lint\|full sweep\|CUDA shell ready\|raco test torch" | tee -a "$HOME/cifar10-resnet/train.log"
echo "done in $(( $(date +%s) - start )) s" | tee -a "$HOME/cifar10-resnet/train.log"
