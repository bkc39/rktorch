import os
os.chdir('/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152')

p = 'torch/nn.rkt'
s = open(p).read()
old = """<<<<<<< HEAD
         binary-cross-entropy-with-logits
         huber-loss
         l1-loss)
=======
         nll-loss)
>>>>>>> origin/master
"""
new = """         binary-cross-entropy-with-logits
         huber-loss
         l1-loss
         nll-loss)
"""
assert old in s
open(p, 'w').write(s.replace(old, new))
print("nn.rkt resolved: both loss sets provided")
