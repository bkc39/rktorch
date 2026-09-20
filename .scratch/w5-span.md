Fair — the previous round checked the endpoints and not the thing the
implementation divides by. The predicate now computes the span and asks
that it be finite and positive:

```racket
(define span (exact->inexact (- (cadr r) (car r))))
(and (< (car r) (cadr r)) (rational? span) (positive? span))
```

`exact->inexact` first, so an exact bignum range degenerating to `0.0`
in `(/ 255.0 span)` is caught the same way, and a pair so close together
that the span underflows to zero fails the `positive?`. Test for
`'(-1e308 1e308)`. Fixed in fbf4db4.
