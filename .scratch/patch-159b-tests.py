import os
os.chdir('/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152')

p = 'torch/tests/ppm-test.rkt'
s = open(p).read()

old = """    (check-exn exn:fail:contract?
               (lambda () (image-grid bytes-batch #:pad-value -1))))"""
new = """    (check-exn #rx"^image-grid: contract violation"
               (lambda () (image-grid bytes-batch #:pad-value -1)))
    (check-exn #rx"uint8-fill-value"
               (lambda () (image-grid bytes-batch #:pad-value 256))))"""
assert old in s
s = s.replace(old, new)

old2 = """  (test-case "image-grid: one image comes back the way make_grid returns it\""""
new2 = """  (test-case "image-grid does not extend the caller's graph, as make_grid"
    (define x (mul (rand 3 3 2 2 #:requires-grad? #t) 1.0))
    (check-true (requires-grad? x))
    (check-false (requires-grad? (image-grid x)))
    (check-false (requires-grad? (image-grid (narrow x 0 0 1)))))

  (test-case "image-grid: one image comes back the way make_grid returns it\""""
assert old2 in s
s = s.replace(old2, new2)

old3 = """    (check-exn #rx"expected: image"
               (lambda () (written (to-dtype (zeros 3 2 2) 'bool)))))"""
new3 = """    (check-exn #rx"expected: image"
               (lambda () (written (to-dtype (zeros 3 2 2) 'bool))))
    ;; a PPM header states a width and a height, and neither may be zero
    (check-exn #rx"expected: image" (lambda () (written (zeros 3 0 2))))
    (check-exn #rx"expected: image" (lambda () (written (zeros 3 2 0)))))"""
assert old3 in s
open(p, 'w').write(s.replace(old3, new3))

p = 'torch/scribblings/vision.scrbl'
s = open(p).read()
old = """@defproc[(image-grid [images tensor?]
                     [#:columns columns exact-positive-integer? 8]
                     [#:padding padding exact-nonnegative-integer? 2]
                     [#:pad-value pad-value real? 0])
         tensor?]{"""
new = """@defproc[(image-grid [images tensor?]
                     [#:columns columns exact-positive-integer? 8]
                     [#:padding padding exact-nonnegative-integer? 2]
                     [#:pad-value pad-value (fill-value/c (tensor-dtype images)) 0])
         tensor?]{"""
assert old in s
s = s.replace(old, new)

old2 = """@tt{make_grid} returns there. The layout is torchvision's
@tt{make_grid}.
}"""
new2 = """@tt{make_grid} returns there. The layout is torchvision's
@tt{make_grid}. @racket[pad-value] must be a value the batch's dtype
holds exactly, so a uint8 batch takes 0 through 255. The grid is built
under @racket[with-no-grad], as @tt{make_grid} is: it is a picture of
the batch, not a step in its graph.
}"""
assert old2 in s
s = s.replace(old2, new2)

old3 = """Writes the @tt{[3 H W]} tensor @racket[image] to @racket[path]."""
new3 = """Writes the @tt{[3 H W]} tensor @racket[image], whose @tt{H} and @tt{W}
are the positive dimensions its header states, to @racket[path]."""
assert old3 in s
open(p, 'w').write(s.replace(old3, new3))
print("159 second-wave tests and docs applied")
