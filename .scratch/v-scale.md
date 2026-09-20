Right, and this is the third form of the same check — endpoints, then
the span, and now the reciprocal that the span never implied. Fixed in
28fe440 by asking about every value the quantization forms rather than
the inputs to it:

```racket
(define (quantizes? r)
  (define span (exact->inexact (- (cadr r) (car r))))
  (and (< (car r) (cadr r))
       (rational? span)
       (positive? span)
       (let ([scale (/ 255.0 span)])
         (and (rational? scale) (positive? scale)))))
```

`'(0 1e-307)` is refused now, with a test.
