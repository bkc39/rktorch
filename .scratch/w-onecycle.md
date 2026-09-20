The NaN is real, but not by the route described, so worth pinning down.
With `#:pct-start 0.1` and `#:total-steps 10`, `up-end` is `0.0` and the
`t = 0` case is `(/ 0 0.0)` — in Racket that is exact `0`, not NaN,
because an exact zero numerator short-circuits:

```
(/ 0 0.0)   = 0
(/ 0.0 0.0) = +nan.0
(/ 1.0 0.0) = +inf.0
```

so that cycle runs clean; I stepped all ten and checked.

`pct-start = 1` is the case that breaks, and for the second reason you
give: `up-end` and `down-end` coincide, the last allowed step divides by
`0.0`, and `+inf.0` through `cos` is NaN. The contract is now
`(and/c (>=/c 0) (</c 1))`, the manual says why, and the test checks both
that 1 is refused and that a short warmup stays finite for every step.
Fixed in 5f93371.
