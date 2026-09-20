import os
os.chdir('/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152')

p = 'torch/tests/ppm-test.rkt'
s = open(p).read()
old = '  (test-case "write-ppm: a P6 header and one byte per channel, row-major"'
new = '''  (test-case "image-grid: one image comes back the way make_grid returns it"
    (define one (reshape (add (arange 12) 1.0) 1 3 2 2))
    (define g (image-grid one #:padding 2 #:pad-value -1))
    (check-equal? (tensor-shape g) '(3 2 2) "no border around a single image")
    (check-equal? (tensor->list g) (tensor->list (select one 0 0)))
    (define grey (reshape (add (arange 4) 1.0) 1 1 2 2))
    (define g1 (image-grid grey #:padding 2))
    (check-equal? (tensor-shape g1) '(3 2 2) "one channel still becomes three")
    (check-equal? (tensor->list (select g1 0 2)) '(1.0 2.0 3.0 4.0)))

  (test-case "write-ppm: a P6 header and one byte per channel, row-major"'''
assert old in s
s = s.replace(old, new)

old2 = """    (check-exn exn:fail:contract?
               (lambda () (written (zeros 3 2 2) #:range '(1 0)))))"""
new2 = """    (check-exn exn:fail:contract?
               (lambda () (written (zeros 3 2 2) #:range '(1 0))))
    (check-exn #rx"expected: image"
               (lambda () (written (to-dtype (zeros 3 2 2) 'bool)))))"""
assert old2 in s
open(p, 'w').write(s.replace(old2, new2))

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
print("tests and docs added")
