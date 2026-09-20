Same finding as the Codex thread just above, and you read the cause
right: the `or` short-circuited on any non-exact-integer.

Fixed in 620b51c with one deliberate difference from the suggested
`(and (exact-integer? value) ...)` — it uses `integer?`, so `2.0` still
crosses. uint8's branch already accepts `255.0`, and
`(full-like (zeros 2) 255.0 #:dtype 'uint8)` is an existing test, so
`exact-integer?` would have made the two dtypes disagree in the other
direction.
