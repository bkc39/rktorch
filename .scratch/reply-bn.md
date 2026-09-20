This does not reproduce: `(batch-norm x #:training? #f)` with no running statistics raises the contract violation, not a native error. I ran every combination against this branch:

| call | result |
|---|---|
| `#:training? #t`, no stats | OK |
| `#:training? #t`, both stats | OK |
| `#:training? #t`, only `#:running-mean` | `batch-norm: contract violation` |
| `#:training? #f` explicit, no stats | `batch-norm: contract violation` |
| `#:training?` absent, no stats | `batch-norm: contract violation` |
| `#:training? #f` explicit, both stats | OK |

The reasoning turns on what `supplied` returns, which is easy to misread. It is `(and (not (unsupplied-arg? v)) v)`, so for a boolean it yields *the value, defaulting to `#f`*, not "was it passed":

- not passed → `#f`
- passed `#t` → `#t`
- passed `#f` → `#f`

So `(if (supplied training?) ...)` tests whether training is *on*, not whether the argument was written, and the two "eval mode" cases — absent and explicit `#f` — take the same else branch, which demands both statistics. The `eq?` pairing check is reached only when training is genuinely on, where ATen does accept no statistics.

You are right that no test pinned the explicit-`#f` case, only the absent one. I will add it to `tensor-ops-test.rkt`'s batch-norm case on the next push to this branch rather than spend a full CI cycle on one assertion now.
