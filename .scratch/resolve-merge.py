"""Resolve the three append-style conflicts from merging vision/ppm-155."""
import re


def hunks(path):
    text = open(path).read()
    parts = []
    pos = 0
    pat = re.compile(r"<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> vision/ppm-155\n", re.S)
    for m in pat.finditer(text):
        parts.append(("text", text[pos:m.start()]))
        parts.append(("hunk", m.group(1), m.group(2)))
        pos = m.end()
    parts.append(("text", text[pos:]))
    return parts


def write(path, parts, resolvers):
    out = []
    k = 0
    for p in parts:
        if p[0] == "text":
            out.append(p[1])
        else:
            out.append(resolvers[k](p[1], p[2]))
            k += 1
    assert k == len(resolvers), (path, k, len(resolvers))
    open(path, "w").write("".join(out))


# AGENTS.md: HEAD's bullets, with theirs' images bullet before the shape line
def agents(head, theirs):
    images = theirs.split("- shape:")[0]
    return head.replace("- shape:", images + "- shape:", 1)


write("AGENTS.md", hunks("AGENTS.md"), [agents])

# vision.scrbl: both for-label lines; both section groups, HEAD's last
# defproc closed before theirs starts
write("torch/scribblings/vision.scrbl", hunks("torch/scribblings/vision.scrbl"),
      [lambda h, t: h + t, lambda h, t: h + "}\n\n" + t])

# python-cross-test: HEAD's imports (a superset); both module requires;
# HEAD's lets, then a fresh let for theirs' block
write("torch/tests/python-cross-test.rkt",
      hunks("torch/tests/python-cross-test.rkt"),
      [lambda h, t: h, lambda h, t: h + t, lambda h, t: h + "     (let ()\n" + t])
print("resolved")
