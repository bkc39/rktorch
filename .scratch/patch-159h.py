import os
os.chdir('/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152')

p = 'torch/vision/ppm.rkt'
s = open(p).read()
old = "(define ppm-dtypes '(float32 float64 float16 bfloat16 uint8))"
new = "(define ppm-dtypes '(float32 float64 uint8))"
assert old in s
open(p, 'w').write(s.replace(old, new))

p = 'torch/scribblings/vision.scrbl'
s = open(p).read()
old = """image is written as it is. @racket[image] is a floating-point or uint8
tensor: an integer or boolean one has no range the transform can read,
and under the default @racket[range] a 0-to-255 integer image would
quantize to white rather than to itself."""
new = """image is written as it is. @racket[image] is a @racket['float32],
@racket['float64] or @racket['uint8] tensor: an integer or boolean one
has no range the transform can read, and under the default
@racket[range] a 0-to-255 integer image would quantize to white rather
than to itself."""
assert old in s
open(p, 'w').write(s.replace(old, new))

# the bool case gets the name its siblings have
p = 'torch/foreign/creation-ops.rkt'
s = open(p).read()
old = """   (case dtype
     [(int64) 'int64-fill-value]
     [(uint8) 'uint8-fill-value]
     [else 'fill-value])"""
new = """   (case dtype
     [(int64) 'int64-fill-value]
     [(uint8) 'uint8-fill-value]
     [(bool) 'bool-fill-value]
     [else 'fill-value])"""
assert old in s
open(p, 'w').write(s.replace(old, new))

p = 'torch/tests/ppm-test.rkt'
s = open(p).read()
old = """    (check-exn #rx"int64-fill-value"
               (lambda () (image-grid int-batch #:pad-value 0.5))))"""
new = """    (check-exn #rx"int64-fill-value"
               (lambda () (image-grid int-batch #:pad-value 0.5)))
    (define bool-batch (to-dtype (zeros 2 3 2 2) 'bool))
    (check-exn #rx"bool-fill-value"
               (lambda () (image-grid bool-batch #:pad-value 2))))"""
assert old in s
open(p, 'w').write(s.replace(old, new))
print("half dtypes dropped, bool-fill-value named")
