import os
os.chdir('/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152')

p = 'torch/vision/ppm.rkt'
s = open(p).read()

# the span, not just the endpoints, has to quantize
old = """(define value-range/c
  (flat-named-contract
   'value-range
   (and/c (list/c rational? rational?) (lambda (r) (< (car r) (cadr r))))))"""
new = """(define value-range/c
  (flat-named-contract
   'value-range
   (and/c (list/c rational? rational?)
          (lambda (r)
            (define span (exact->inexact (- (cadr r) (car r))))
            (and (< (car r) (cadr r)) (rational? span) (positive? span))))))"""
assert old in s
s = s.replace(old, new)

# the manual carries this one
old2 = """;; make_grid returns a single image as it is, with no border
(define (one-image image c h w)"""
new2 = """(define (one-image image c h w)"""
assert old2 in s
s = s.replace(old2, new2)

# save_image quantizes under no-grad too
old3 = """  (define pixels
    (tensor->vector
     (permute (if (eq? (tensor-dtype image) 'uint8)
                  image
                  (to-dtype (clamp (add (mul (sub image lo) (/ 255.0 (- hi lo)))
                                        0.5)
                                   #:min 0 #:max 255)
                            'uint8))
              1 2 0)))"""
new3 = """  (define pixels
    (with-no-grad
      (tensor->vector
       (permute (if (eq? (tensor-dtype image) 'uint8)
                    image
                    (to-dtype (clamp (add (mul (sub image lo)
                                               (/ 255.0 (- hi lo)))
                                          0.5)
                                     #:min 0 #:max 255)
                              'uint8))
                1 2 0))))"""
assert old3 in s
s = s.replace(old3, new3)
open(p, 'w').write(s)

# tests
p = 'torch/tests/ppm-test.rkt'
s = open(p).read()
old = """    (check-exn #rx"value-range"
               (lambda () (written (zeros 3 2 2) #:range '(-inf.0 1))))"""
new = """    (check-exn #rx"value-range"
               (lambda () (written (zeros 3 2 2) #:range '(-inf.0 1))))
    ;; finite endpoints whose span is not: the scale would come out zero
    (check-exn #rx"value-range"
               (lambda () (written (zeros 3 2 2) #:range '(-1e308 1e308))))"""
assert old in s
s = s.replace(old, new)

old2 = """  (test-case "image-grid and write-ppm accept device tensors\""""
new2 = """  (test-case "write-ppm quantizes off the graph, as save_image does"
    (define x (mul (rand 3 2 2 #:requires-grad? #t) 1.0))
    (check-true (requires-grad? x))
    (define path (make-temporary-file "rkt-grad-~a.ppm"))
    (write-ppm path x)
    (check-equal? (bytes-length (file->bytes path)) (+ 11 (* 3 2 2)))
    (delete-file path))

  (test-case "image-grid and write-ppm accept device tensors\""""
assert old2 in s
open(p, 'w').write(s.replace(old2, new2))
print("span, comment and no-grad quantization applied")
