You are right, and the manual I wrote last round is what makes it a
defect rather than an omission: it promises "no zero dimension" and the
predicate only asked about the last two.

The helper is gone rather than extended — once every dimension has to be
positive, the whole shape is the check:

```racket
(and/c image-batch/c (lambda (x) (andmap positive? (tensor-shape x))))
```

and `image/c` does the same over its `[3 H W]`. `(zeros 2 0 4 4)` is
contract blame now, with a test beside the N, H and W ones. Fixed in
bc21466.
