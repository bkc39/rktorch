import os
os.chdir('/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152')

p = 'AGENTS.md'
s = open(p).read()
old = """<<<<<<< HEAD
sgd adam rmsprop step! zero-grads! learning-rate set-learning-rate! step-lr
multi-step-lr exponential-lr cosine-annealing-lr linear-lr one-cycle-lr
lambda-lr ema ema-update! ema-average cross-entropy
mse-loss binary-cross-entropy-with-logits huber-loss l1-loss kaiming-uniform
uniform-init normal-init fan-in`. The functional
=======
LSTM GRU sgd adam step! zero-grads! clip-grad-norm! ema ema-update! ema-average
cross-entropy nll-loss mse-loss binary-cross-entropy-with-logits huber-loss
l1-loss kaiming-uniform uniform-init normal-init fan-in`. The functional
>>>>>>> vision/half-152
"""
new = """LSTM GRU sgd adam rmsprop step! zero-grads! clip-grad-norm! learning-rate
set-learning-rate! step-lr multi-step-lr exponential-lr cosine-annealing-lr
linear-lr one-cycle-lr lambda-lr ema ema-update! ema-average cross-entropy
nll-loss mse-loss binary-cross-entropy-with-logits huber-loss l1-loss
kaiming-uniform uniform-init normal-init fan-in`. The functional
"""
assert old in s
open(p, 'w').write(s.replace(old, new))
print("AGENTS.md resolved on optim-152")
