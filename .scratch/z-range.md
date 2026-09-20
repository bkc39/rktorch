Confirmed: `'(0 +inf.0)` passed, `(/ 255.0 (- hi lo))` became `0.0`, and
every pixel was written as 0 — the range endpoints mapping to 0 and 255
is exactly what the manual promises.

`value-range/c` asks for `rational?` now instead of `real?`, which
excludes the infinities and NaN together. NaN was already unreachable
through the `(< lo hi)` test, but it costs nothing to have one predicate
mean it. Tests for both infinities. Fixed in bc21466.
