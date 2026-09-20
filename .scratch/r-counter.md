Confirmed, and it is worse than the cast: the promotion happens in `add`,
before `to-dtype` sees it, so the counter was already float32 when it was
cast back. At 2^24 the sum is the same float as the input:

```
scalar add -> dtype float32, value 16777216
tensor add -> dtype int64, value 16777217
```

`(add num-batches-tracked (ones-like num-batches-tracked))` keeps the
int64 dtype through the add, so the `to-dtype` is gone too. There is a
regression test in nn-test.rkt that sets the buffer to 2^24 and checks
the next forward reaches 2^24 + 1 in int64. Fixed in 97d8279.
