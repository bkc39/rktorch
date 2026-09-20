p = '/home/bkc/.claude/projects/-home-bkc/memory/MEMORY.md'
s = open(p).read()
anchor = "- [rktorch next arcs #152 #153](rktorch-next-arcs-152-153.md)"
line = ("- [rktorch vision arc #152](rktorch-vision-arc-152.md) — five stacked "
        "PRs #158 #165 #167 #170 #171 plus #159 (ppm) and #179 (audio flake); "
        "ResNet-18 94.1% on CIFAR-10; one open thread: write-ppm's dtype "
        "contract, needs the owner's call\n")
i = s.index(anchor)
end = s.index("\n", i) + 1
assert line not in s
open(p, 'w').write(s[:end] + line + s[end:])
print("index line added")
