import os
os.chdir('/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152')

# ---- the fill rule becomes a contract both boundaries can state
p = 'torch/foreign/creation-ops.rkt'
s = open(p).read()
old = """(define/contract-out (full value #:device [device #f] #:dtype [dtype #f]"""
new = """(define/contract-out (fill-value/c dtype) ;; noqa
  (-> dtype/c flat-contract?)
  (flat-named-contract
   (case dtype
     [(int64) 'int64-fill-value]
     [(uint8) 'uint8-fill-value]
     [else 'fill-value])
   (lambda (v) (and (real? v) (eq? #t (fill-crosses-exactly? v dtype))))))

(define/contract-out (full value #:device [device #f] #:dtype [dtype #f]"""
assert old in s
s = s.replace(old, new, 1)
open(p, 'w').write(s)

# ---- ppm.rkt: no-grad assembly, a dtype-aware pad value, positive dimensions
p = 'torch/vision/ppm.rkt'
s = open(p).read()

old = """(require (only-in racket/contract/base
                  ->* and/c flat-named-contract list/c)
         (only-in "../foreign.rkt"
                  add clamp copy! full mul narrow permute select sub
                  tensor-device tensor-dtype tensor-shape tensor->vector
                  tensor? to-dtype)"""
new = """(require (only-in racket/contract/base
                  ->* ->i and/c flat-named-contract list/c)
         (only-in "../foreign.rkt"
                  add clamp copy! fill-value/c full mul narrow permute select
                  sub tensor-device tensor-dtype tensor-shape tensor->vector
                  tensor? to-dtype with-no-grad)"""
assert old in s
s = s.replace(old, new)

old = """;; ATen has no subtraction on a boolean tensor, so the range transform
;; has nothing to apply there
(define image/c
  (flat-named-contract
   'image
   (lambda (x)
     (and (tensor? x)
          (not (eq? (tensor-dtype x) 'bool))
          (let ([dims (tensor-shape x)])
            (and (= 3 (length dims)) (= 3 (car dims))))))))"""
new = """;; ATen has no subtraction on a boolean tensor, so the range transform
;; has nothing to apply there; a PPM states its width and height, and
;; neither may be zero
(define image/c
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
assert old in s
s = s.replace(old, new)

old = """(define/contract-out (image-grid images ;; noqa
                                 #:columns [columns 8]
                                 #:padding [padding 2]
                                 #:pad-value [pad-value 0])
  (->* [non-empty-image-batch/c]
       [#:columns exact-positive-integer?
        #:padding exact-nonnegative-integer?
        #:pad-value real?]
       tensor?)
  (define dims (tensor-shape images))"""
new = """(define/contract-out (image-grid images ;; noqa
                                 #:columns [columns 8]
                                 #:padding [padding 2]
                                 #:pad-value [pad-value 0])
  (->i ([images non-empty-image-batch/c])
       (#:columns [columns exact-positive-integer?]
        #:padding [padding exact-nonnegative-integer?]
        #:pad-value [pad-value (images) (fill-value/c (tensor-dtype images))])
       [result tensor?])
  (define dims (tensor-shape images))"""
assert old in s
s = s.replace(old, new)

old = """  (cond
    [(= n 1) (one-image (select images 0 0) c h w)]
    [else (grid-of images n c h w columns padding pad-value)]))"""
new = """  ;; make_grid is a display helper and carries @torch.no_grad(); without
  ;; it every copy! would extend the caller's graph into the grid
  (with-no-grad
    (cond
      [(= n 1) (one-image (select images 0 0) c h w)]
      [else (grid-of images n c h w columns padding pad-value)])))"""
assert old in s
s = s.replace(old, new)
open(p, 'w').write(s)
print("159 second-wave patch applied")
