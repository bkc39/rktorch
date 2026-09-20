import os
os.chdir('/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152')
p = 'torch/scribblings/vision.scrbl'
s = open(p).read()
old = """under @racket[with-no-grad], as @tt{make_grid} carries
@tt{@@torch.no_grad()}: it is a picture of the batch, not a step in
its graph."""
new = """under @racket[with-no-grad], as @tt{make_grid} is decorated with
@tt{no_grad}: it is a picture of the batch, not a step in its graph."""
assert old in s
open(p, 'w').write(s.replace(old, new))
print("escaped")
