import os
os.chdir('/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152')

# ---- int64 gets the integer check uint8 already has
p = 'torch/foreign/creation-ops.rkt'
s = open(p).read()
old = """    [(eq? dtype 'int64)
     (or (not (exact-integer? value))
         (= (exact->inexact value) value)
         "an int64 fill value must be exactly representable as a double")]"""
new = """    [(eq? dtype 'int64)
     (or (and (integer? value) (= (exact->inexact value) value))
         "an int64 fill value must be an integer a double holds exactly")]"""
assert old in s
s = s.replace(old, new)

old2 = """;; the fill crosses the FFI as a double: an int64 fill outside the exact
;; range of a double would round silently and a uint8 fill outside 0..255
;; would wrap, so the contract refuses both"""
new2 = """;; the fill crosses the FFI as a double: a fractional value would truncate,
;; an int64 fill outside the exact range of a double would round and a
;; uint8 fill outside 0..255 would wrap, so the contract refuses all three
;; where torch takes them silently"""
assert old2 in s
open(p, 'w').write(s.replace(old2, new2))

# ---- the comment the manual already carries
p = 'torch/vision/ppm.rkt'
s = open(p).read()
old = """;; ATen has no subtraction on a boolean tensor, so the range transform
;; has nothing to apply there; a PPM states its width and height, and
;; neither may be zero
(define image/c"""
new = """(define image/c"""
assert old in s
open(p, 'w').write(s.replace(old, new))

# ---- tests
p = 'torch/tests/bytes-ingestion-test.rkt'
s = open(p).read()
old = """    (check-exn #rx"uint8 fill value must be an integer from 0 to 255"
               (lambda () (full-like (zeros 2) -1 #:dtype 'uint8)))"""
new = """    (check-exn #rx"uint8 fill value must be an integer from 0 to 255"
               (lambda () (full-like (zeros 2) -1 #:dtype 'uint8)))
    ;; torch truncates a fractional fill for both dtypes; int64 refuses it
    ;; here for the reason uint8 does
    (check-equal? (tensor->list (full 2.0 2 #:dtype 'int64)) '(2 2))
    (check-exn #rx"int64 fill value must be an integer"
               (lambda () (full 0.5 2 #:dtype 'int64)))
    (check-exn #rx"int64 fill value must be an integer"
               (lambda () (full 1/2 2 #:dtype 'int64)))
    (check-exn #rx"int64 fill value must be an integer"
               (lambda () (full (add1 (expt 2 60)) 2 #:dtype 'int64)))"""
assert old in s
open(p, 'w').write(s.replace(old, new))

p = 'torch/tests/ppm-test.rkt'
s = open(p).read()
old = """    (check-exn #rx"uint8-fill-value"
               (lambda () (image-grid bytes-batch #:pad-value 256))))"""
new = """    (check-exn #rx"uint8-fill-value"
               (lambda () (image-grid bytes-batch #:pad-value 256)))
    (define int-batch (to-dtype (full 9.0 2 3 2 2) 'int64))
    (check-exn #rx"^image-grid: contract violation"
               (lambda () (image-grid int-batch #:pad-value 0.5)))
    (check-exn #rx"int64-fill-value"
               (lambda () (image-grid int-batch #:pad-value 0.5))))"""
assert old in s
open(p, 'w').write(s.replace(old, new))
print("int64 fill tightened, comment removed, tests added")
