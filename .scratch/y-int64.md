Confirmed and fixed in 620b51c. The two dtypes disagreed in one
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
