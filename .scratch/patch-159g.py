import os
os.chdir('/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152')

p = 'torch/vision/ppm.rkt'
s = open(p).read()
old = """(define value-range/c
  (flat-named-contract
   'value-range
   (and/c (list/c rational? rational?)
          (lambda (r)
            (define span (exact->inexact (- (cadr r) (car r))))
            (and (< (car r) (cadr r)) (rational? span) (positive? span))))))"""
new = """(define (quantizes? r)
  (define span (exact->inexact (- (cadr r) (car r))))
  (and (< (car r) (cadr r))
       (rational? span)
       (positive? span)
       (let ([scale (/ 255.0 span)])
         (and (rational? scale) (positive? scale)))))

(define value-range/c
  (flat-named-contract
   'value-range
   (and/c (list/c rational? rational?) quantizes?)))"""
assert old in s
open(p, 'w').write(s.replace(old, new))

p = 'torch/tests/ppm-test.rkt'
s = open(p).read()
old = """    (check-exn #rx"value-range"
               (lambda () (written (zeros 3 2 2) #:range '(-1e308 1e308))))"""
new = """    (check-exn #rx"value-range"
               (lambda () (written (zeros 3 2 2) #:range '(-1e308 1e308))))
    ;; a span small enough that 255 over it is not a number either
    (check-exn #rx"value-range"
               (lambda () (written (zeros 3 2 2) #:range '(0 1e-307))))"""
assert old in s
open(p, 'w').write(s.replace(old, new))
print("scale check applied")
