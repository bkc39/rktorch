Right — the error already named the constraint, but it named `full` and
blamed `torch/vision/ppm.rkt` for what the caller passed, which is
backwards for a library.

The dtype rule now has a name. `creation-ops.rkt` exports
`(fill-value/c dtype)`, built from the same table `full`'s `#:pre/desc`
uses, so there is still one place that knows an int64 fill must be exact
in a double and a uint8 fill must be 0 through 255. `image-grid` takes
`->i` and states it:

```racket
  #:pad-value [pad-value (images) (fill-value/c (tensor-dtype images))]
```

so `(image-grid uint8-batch #:pad-value -1)` is now `image-grid: contract
violation` blaming the caller, with `uint8-fill-value` as the expected
contract. Tested both below and above the range.
