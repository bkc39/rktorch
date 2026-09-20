import os
os.chdir('/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152')

p = 'AGENTS.md'
s = open(p).read()

old = """<<<<<<< HEAD
  `Sequential`, `Embedding`, `LayerNorm`, `GroupNorm`, `BatchNorm2d`,
  `BatchNorm1d`), mirroring the
=======
  `Sequential`, `Embedding`, `LayerNorm`, `GroupNorm`, `LSTM`, `GRU`),
  mirroring the
>>>>>>> origin/master
"""
new = """  `Sequential`, `Embedding`, `LayerNorm`, `GroupNorm`, `BatchNorm2d`,
  `BatchNorm1d`, `LSTM`, `GRU`), mirroring the
"""
assert old in s
s = s.replace(old, new)

# master named LSTM and GRU as constructors without adding their predicates
old_pred = """  `sequential?`, `embedding?`, `layer-norm?`, `group-norm?`,
  `batch-norm2d?`, `batch-norm1d?`), per Racket idiom"""
new_pred = """  `sequential?`, `embedding?`, `layer-norm?`, `group-norm?`,
  `batch-norm2d?`, `batch-norm1d?`, `lstm?`, `gru?`), per Racket idiom"""
assert old_pred in s
s = s.replace(old_pred, new_pred)

old2 = """<<<<<<< HEAD
Sequential Embedding LayerNorm ConvTranspose2d GroupNorm BatchNorm2d BatchNorm1d
sgd adam step! zero-grads! ema ema-update! ema-average cross-entropy
mse-loss binary-cross-entropy-with-logits huber-loss l1-loss kaiming-uniform
uniform-init normal-init fan-in`. The functional
=======
Sequential Embedding LayerNorm ConvTranspose2d GroupNorm LSTM GRU sgd adam step!
zero-grads! clip-grad-norm! ema
ema-update! ema-average cross-entropy nll-loss
mse-loss kaiming-uniform uniform-init normal-init fan-in`. The functional
>>>>>>> origin/master
"""
new2 = """Sequential Embedding LayerNorm ConvTranspose2d GroupNorm BatchNorm2d BatchNorm1d
LSTM GRU sgd adam step! zero-grads! clip-grad-norm! ema ema-update! ema-average
cross-entropy nll-loss mse-loss binary-cross-entropy-with-logits huber-loss
l1-loss kaiming-uniform uniform-init normal-init fan-in`. The functional
"""
assert old2 in s
open(p, 'w').write(s.replace(old2, new2))
print("AGENTS.md resolved: both arcs' names kept")
