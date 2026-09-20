Good catch, and it is the more interesting half of the pair: `image/c`
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
