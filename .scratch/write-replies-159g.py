import os
os.chdir('/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152/.scratch')

replies = {
'v-int64promo.md': """This one is overtaken by 42f6a66, which landed while the review was
computing against fbf4db4: the owner's call was to restrict the dtypes,
so `image/c` now accepts only float32, float64, float16, bfloat16 and
uint8. An int64 image never reaches the range arithmetic — it is
contract blame at the boundary, with a test.

The analysis was right for the code it read, and it is the same
mechanism as the BatchNorm counter earlier in this arc: an integer
tensor combined with a floating scalar promotes and rounds. Worth
recording here in case an integer path is ever wanted; it would need
float64 arithmetic, not float32.
""",

'v-scale.md': """Right, and this is the third form of the same check — endpoints, then
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
""",

'v-booltensor.md': """Good pointer, but it no longer applies: 42f6a66 replaced the inline
boolean exclusion with a positive allowlist, since the owner's call was
to restrict the dtypes rather than subtract one.

```racket
(define ppm-dtypes '(float32 float64 float16 bfloat16 uint8))
```

so there is no "is this tensor boolean" predicate here to drift from
`bool-tensor/c` — bool is simply absent from the list, along with int64.
""",
}

for name, body in replies.items():
    open(name, 'w').write(body)
print("wrote", len(replies), "replies")
