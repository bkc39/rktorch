import os
os.chdir('/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152')

# the message has to serve both the fractional case and the 2^53 one
p = 'torch/foreign/creation-ops.rkt'
s = open(p).read()
old = '"an int64 fill value must be an integer a double holds exactly"'
new = '"an int64 fill value must be an integer exactly representable as a double"'
assert old in s
open(p, 'w').write(s.replace(old, new))

# the batch contract gets image/c's positive-pixel check
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
                 (positive? (cadr dims))
                 (positive? (caddr dims))))))))"""
new = """(define (pixels? dims)
  (andmap positive? (list-tail dims (- (length dims) 2))))

(define image/c
  (flat-named-contract
   'image
   (lambda (x)
     (and (tensor? x)
          (not (eq? (tensor-dtype x) 'bool))
          (let ([dims (tensor-shape x)])
            (and (= 3 (length dims)) (= 3 (car dims)) (pixels? dims)))))))"""
assert old in s
s = s.replace(old, new)

old2 = """(define non-empty-image-batch/c
  (flat-named-contract
   'non-empty-image-batch
   (and/c image-batch/c (lambda (x) (positive? (car (tensor-shape x)))))))"""
new2 = """(define non-empty-image-batch/c
  (flat-named-contract
   'non-empty-image-batch
   (and/c image-batch/c
          (lambda (x)
            (let ([dims (tensor-shape x)])
              (and (positive? (car dims)) (pixels? dims)))))))"""
assert old2 in s
open(p, 'w').write(s.replace(old2, new2))

# tests
p = 'torch/tests/ppm-test.rkt'
s = open(p).read()
old = """    (check-exn #rx"non-empty-image-batch"
               (lambda () (image-grid (zeros 3 4 4))))"""
new = """    (check-exn #rx"non-empty-image-batch"
               (lambda () (image-grid (zeros 3 4 4))))
    ;; a zero height or width makes a grid that is nothing but padding
    (check-exn #rx"non-empty-image-batch"
               (lambda () (image-grid (zeros 2 3 0 5))))
    (check-exn #rx"non-empty-image-batch"
               (lambda () (image-grid (zeros 2 3 5 0))))"""
assert old in s
open(p, 'w').write(s.replace(old, new))

# the manual says what the batch must be
p = 'torch/scribblings/vision.scrbl'
s = open(p).read()
old = """least one image, out as one @tt{[C H' W']} image, @racket[columns] across"""
new = """least one image and no zero dimension, out as one @tt{[C H' W']} image,
@racket[columns] across"""
assert old in s
open(p, 'w').write(s.replace(old, new))
print("message reworded, batch contract shares the pixel check")
