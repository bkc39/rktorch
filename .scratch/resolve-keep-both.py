import re, sys
# keep both sides of an append-style conflict, whatever the incoming ref is
pat = re.compile(r"<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n", re.S)
for path in sys.argv[1:]:
    s = open(path).read()
    out, n = pat.subn(lambda m: m.group(1) + m.group(2), s)
    open(path, 'w').write(out)
    print(f"  {path}: {n} hunk(s)")
