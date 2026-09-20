import os
os.chdir('/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152')

# ---- every dimension, not just the last two; and a finite range
p = 'torch/vision/ppm.rkt'
s = open(p).read()
old = """(define (pixels? dims)
  (andmap positive? (list-tail dims (- (length dims) 2))))

(define image/c
  (flat-named-contract
   'image
   (lambda (x)
     (and (tensor? x)
          (not (eq? (tensor-dtype x) 'bool))
          (let ([dims (tensor-shape x)])
            (and (= 3 (length dims)) (= 3 (car dims)) (pixels? dims)))))))"""
new = """(define image/c
  (flat-named-contract
   'image
   (lambda (x)
     (and (tensor? x)
          (not (eq? (tensor-dtype x) 'bool))
          (let ([dims (tensor-shape x)])
            (and (= 3 (length dims))
                 (= 3 (car dims))
                 (andmap positive? dims)))))))"""
assert old in s
s = s.replace(old, new)

old2 = """(define value-range/c
  (flat-named-contract
   'value-range
   (and/c (list/c real? real?) (lambda (r) (< (car r) (cadr r))))))"""
new2 = """(define value-range/c
  (flat-named-contract
   'value-range
   (and/c (list/c rational? rational?) (lambda (r) (< (car r) (cadr r))))))"""
assert old2 in s
s = s.replace(old2, new2)

old3 = """(define non-empty-image-batch/c
  (flat-named-contract
   'non-empty-image-batch
   (and/c image-batch/c
          (lambda (x)
            (let ([dims (tensor-shape x)])
              (and (positive? (car dims)) (pixels? dims)))))))"""
new3 = """(define non-empty-image-batch/c
  (flat-named-contract
   'non-empty-image-batch
   (and/c image-batch/c (lambda (x) (andmap positive? (tensor-shape x))))))"""
assert old3 in s
s = s.replace(old3, new3)
open(p, 'w').write(s)

# ---- int64 bounds and a boolean case
p = 'torch/foreign/creation-ops.rkt'
s = open(p).read()
old = """    [(eq? dtype 'int64)
     (or (and (integer? value) (= (exact->inexact value) value))
         "an int64 fill value must be an integer exactly representable as a double")]
    [(eq? dtype 'uint8)
     (or (and (integer? value) (<= 0 value 255))
         "a uint8 fill value must be an integer from 0 to 255")]
    [else #t]))"""
new = """    [(eq? dtype 'int64)
     (or (and (integer? value)
              (<= (- (expt 2 63)) value (sub1 (expt 2 63)))
              (= (exact->inexact value) value))
         "an int64 fill value must be an integer exactly representable as a double")]
    [(eq? dtype 'uint8)
     (or (and (integer? value) (<= 0 value 255))
         "a uint8 fill value must be an integer from 0 to 255")]
    [(eq? dtype 'bool)
     (or (and (integer? value) (<= 0 value 1))
         "a bool fill value must be 0 or 1")]
    [else #t]))"""
assert old in s
open(p, 'w').write(s.replace(old, new))

# ---- tests
p = 'torch/tests/ppm-test.rkt'
s = open(p).read()
old = """    (check-exn #rx"non-empty-image-batch"
               (lambda () (image-grid (zeros 2 3 5 0))))"""
new = """    (check-exn #rx"non-empty-image-batch"
               (lambda () (image-grid (zeros 2 3 5 0))))
    (check-exn #rx"non-empty-image-batch"
               (lambda () (image-grid (zeros 2 0 4 4))))"""
assert old in s
s = s.replace(old, new)

old2 = """    (check-exn exn:fail:contract?
               (lambda () (written (zeros 3 2 2) #:range '(1 0))))"""
new2 = """    (check-exn exn:fail:contract?
               (lambda () (written (zeros 3 2 2) #:range '(1 0))))
    ;; an infinite span makes the scale zero and writes every pixel black
    (check-exn #rx"value-range"
               (lambda () (written (zeros 3 2 2) #:range '(0 +inf.0))))
    (check-exn #rx"value-range"
               (lambda () (written (zeros 3 2 2) #:range '(-inf.0 1))))"""
assert old2 in s
open(p, 'w').write(s.replace(old2, new2))

p = 'torch/tests/bytes-ingestion-test.rkt'
s = open(p).read()
old = """    (check-exn #rx"int64 fill value must be an integer"
               (lambda () (full (add1 (expt 2 60)) 2 #:dtype 'int64)))"""
new = """    (check-exn #rx"int64 fill value must be an integer"
               (lambda () (full (add1 (expt 2 60)) 2 #:dtype 'int64)))
    (check-exn #rx"int64 fill value must be an integer"
               (lambda () (full (expt 2 63) 2 #:dtype 'int64)))
    (check-equal? (tensor->list (full 1 2 #:dtype 'bool)) '(#t #t))
    (check-exn #rx"bool fill value must be 0 or 1"
               (lambda () (full 0.5 2 #:dtype 'bool)))"""
assert old in s
open(p, 'w').write(s.replace(old, new))

# ---- docs
p = 'torch/scribblings/vision.scrbl'
s = open(p).read()
old = """its second, so a dataset in @tt{[-1, 1]} passes @racket['(-1 1)]; a uint8"""
new = """its second and both finite, so a dataset in @tt{[-1, 1]} passes
@racket['(-1 1)]; a uint8"""
assert old in s
open(p, 'w').write(s.replace(old, new))
print("channel, bounds, bool and finite-range checks applied")
