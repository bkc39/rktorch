import os
os.chdir('/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152')

p = 'torch/nn/loss.rkt'
s = open(p).read()

old = """<<<<<<< HEAD
(require (only-in racket/contract/base -> ->* >/c and/c listof or/c)
=======
(require (only-in racket/contract/base -> ->* and/c listof or/c)
>>>>>>> origin/master
"""
new = """(require (only-in racket/contract/base -> ->* >/c and/c listof or/c)
"""
assert old in s
s = s.replace(old, new)

old2 = """<<<<<<< HEAD
                  [huber-loss g:huber-loss]
                  [l1-loss g:l1-loss])
=======
                  [nll-loss g:nll-loss])
>>>>>>> origin/master
"""
new2 = """                  [huber-loss g:huber-loss]
                  [l1-loss g:l1-loss]
                  [nll-loss g:nll-loss])
"""
assert old2 in s
open(p, 'w').write(s.replace(old2, new2))
print("loss.rkt resolved: both sides' imports kept, alphabetized")
