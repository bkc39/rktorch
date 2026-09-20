import os
os.chdir('/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152/.scratch')

replies = {
'y-int64.md': """Confirmed and fixed in 620b51c. The two dtypes disagreed in one
predicate:

```
full 0.5 int64  -> (0 0)
full 0.5 uint8  -> RAISED: full: contract violation
```

Both require an integer now. One deliberate difference from the
suggestion: `integer?`, not `exact-integer?`, so `(full 2.0 2 #:dtype
'int64)` still works — uint8 already accepts `255.0` the same way, and
the existing `full-like` test relies on that. The message keeps the
phrase the 2^53 goldens match, so it reads for both cases: "an int64
fill value must be an integer exactly representable as a double".

Worth recording that torch truncates for **both** dtypes —
`torch.full((2,), 0.5, dtype=torch.uint8)` is `[0, 0]` there too — so
refusing is a departure this table had already made for uint8 and has
now made consistently.
""",

'y-comment.md': """Agreed — vision.scrbl carries both halves already ("A boolean image is
not one ATen can subtract a range from, so the contract refuses it", and
the positive-dimension sentence), so the comment was the manual repeated
in the source. Removed in 620b51c.
""",

'y-batchdims.md': """Good catch, and it is the more interesting half of the pair: `image/c`
rejected a zero dimension while `non-empty-image-batch/c` did not, so
`(zeros 2 3 0 5)` reached `image-grid` and came back as a grid of nothing
but padding, with no error anywhere unless it later met `write-ppm`.

Factored out, as you suggest, rather than duplicated:

```racket
(define (pixels? dims)
  (andmap positive? (list-tail dims (- (length dims) 2))))
```

used by both contracts, with tests for a zero height and a zero width.
Fixed in 620b51c.
""",

'y-int64-dup.md': """Same finding as the Codex thread just above, and you read the cause
right: the `or` short-circuited on any non-exact-integer.

Fixed in 620b51c with one deliberate difference from the suggested
`(and (exact-integer? value) ...)` — it uses `integer?`, so `2.0` still
crosses. uint8's branch already accepts `255.0`, and
`(full-like (zeros 2) 255.0 #:dtype 'uint8)` is an existing test, so
`exact-integer?` would have made the two dtypes disagree in the other
direction.
""",
}

for name, body in replies.items():
    open(name, 'w').write(body)
print("wrote", len(replies), "replies")
