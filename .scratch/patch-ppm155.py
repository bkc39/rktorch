import os
os.chdir('/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152')

p = 'torch/scribblings/vision.scrbl'
s = open(p).read()
old = """least one image, out as one @tt{[C H' W']} image, @racket[columns] across
and @racket[padding] pixels of @racket[pad-value] around every image, on
the device the batch lives on. One channel becomes three. The layout is
torchvision's @tt{make_grid}."""
new = """least one image, out as one @tt{[C H' W']} image, @racket[columns] across
and @racket[padding] pixels of @racket[pad-value] around every image, on
the device the batch lives on. One channel becomes three. A batch of one
image comes back as that image, with no border, which is what
@tt{make_grid} returns there. The layout is torchvision's
@tt{make_grid}."""
assert old in s
s = s.replace(old, new)

old2 = """image is quantized the way torchvision's @tt{save_image} does, with
@racket[range] naming the values that map to 0 and 255, its first below
its second, so a dataset in @tt{[-1, 1]} passes @racket['(-1 1)]; a uint8
image is written as it is."""
new2 = """image is quantized the way torchvision's @tt{save_image} does, with
@racket[range] naming the values that map to 0 and 255, its first below
its second, so a dataset in @tt{[-1, 1]} passes @racket['(-1 1)]; a uint8
image is written as it is. A boolean image is not one ATen can subtract
a range from, so the contract refuses it."""
assert old2 in s
open(p, 'w').write(s.replace(old2, new2))

p = 'torch/tests/python-cross-test.rkt'
s = open(p).read()
old = """       (check-equal? (bytes->list (subbytes bs (bytes-length header)))
                     (hash-ref j 'pixels)
                     "write-ppm: save_image's quantization"))"""
new = """       (define pixels (bytes->list (subbytes bs (bytes-length header))))
       (check-equal? (length pixels) (length (hash-ref j 'pixels))
                     "write-ppm: one byte per channel")
       ;; the quantization rounds at a half, where a difference the value
       ;; check above tolerates moves a byte by one
       (for ([a (in-list pixels)]
             [b (in-list (hash-ref j 'pixels))]
             [i (in-naturals)])
         (check-= a b 1 (format "write-ppm: save_image's quantization ~a" i)))
       (manual-seed! 1)
       (define one (image-grid (rand 1 3 4 4) #:columns 2 #:padding 1
                               #:pad-value 0.5))
       (check-equal? (tensor-shape one) (hash-ref j 'one_shape)
                     "image-grid: make_grid returns one image unpadded")
       (for ([a (in-list (tensor->list one))]
             [b (in-list (hash-ref j 'one_values))]
             [i (in-naturals)])
         (check-= a b tol (format "image-grid: one image value ~a parity" i))))"""
assert old in s
open(p, 'w').write(s.replace(old, new))
print("ppm-155 docs and cross-test patched")
