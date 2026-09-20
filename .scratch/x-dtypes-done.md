The owner's call is to restrict, so this is done in 42f6a66. `image/c`
names the dtypes it accepts:

```racket
(define ppm-dtypes '(float32 float64 float16 bfloat16 uint8))
```

An int64 or boolean image is contract blame at the boundary now instead
of a white picture or an ATen error, with a test for each.

One thing beyond the literal ask: the list carries float16 and bfloat16,
which this branch cannot produce — `dtype/c` here is float32, float64,
int64, bool and uint8. #165 adds the half dtypes in this same arc, and a
sample tensor from an autocast forward would otherwise start failing the
moment that merges, on a branch nobody would think to re-check. The two
symbols are inert until then.
