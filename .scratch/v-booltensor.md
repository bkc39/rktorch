Good pointer, but it no longer applies: 42f6a66 replaced the inline
boolean exclusion with a positive allowlist, since the owner's call was
to restrict the dtypes rather than subtract one.

```racket
(define ppm-dtypes '(float32 float64 float16 bfloat16 uint8))
```

so there is no "is this tensor boolean" predicate here to drift from
`bool-tensor/c` — bool is simply absent from the list, along with int64.
