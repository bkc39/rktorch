import os
os.chdir('/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152')

p = 'torch/vision/ppm.rkt'
s = open(p).read()
old = """(define image/c
  (flat-named-contract
   'image
   (lambda (x)
     (and (tensor? x)
          (not (eq? (tensor-dtype x) 'bool))
          (let ([dims (tensor-shape x)])
            (and (= 3 (length dims))
                 (= 3 (car dims))
                 (andmap positive? dims)))))))"""
new = """(define ppm-dtypes '(float32 float64 float16 bfloat16 uint8))

(define image/c
  (flat-named-contract
   'image
   (lambda (x)
     (and (tensor? x)
          (and (memq (tensor-dtype x) ppm-dtypes) #t)
          (let ([dims (tensor-shape x)])
            (and (= 3 (length dims))
                 (= 3 (car dims))
                 (andmap positive? dims)))))))"""
assert old in s
open(p, 'w').write(s.replace(old, new))

p = 'torch/tests/ppm-test.rkt'
s = open(p).read()
old = """    (check-exn #rx"expected: image"
               (lambda () (written (to-dtype (zeros 3 2 2) 'bool))))"""
new = """    (check-exn #rx"expected: image"
               (lambda () (written (to-dtype (zeros 3 2 2) 'bool))))
    ;; an int64 image under the default range would quantize to white
    (check-exn #rx"expected: image"
               (lambda () (written (to-dtype (zeros 3 2 2) 'int64))))"""
assert old in s
open(p, 'w').write(s.replace(old, new))

p = 'torch/scribblings/vision.scrbl'
s = open(p).read()
old = """image is written as it is. A boolean image is not one ATen can subtract
a range from, so the contract refuses it."""
new = """image is written as it is. @racket[image] is a floating-point or uint8
tensor: an integer or boolean one has no range the transform can read,
and under the default @racket[range] a 0-to-255 integer image would
quantize to white rather than to itself."""
assert old in s
open(p, 'w').write(s.replace(old, new))
print("dtypes restricted")
