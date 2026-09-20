import io, os
os.chdir('/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152')

# ---- G: the dcgan targets follow the device the step was given
p='examples/racket/10-dcgan.rkt'
s=open(p).read()
old="""  (define ones-target (ones n 1))
  (define zeros-target (zeros n 1))"""
new="""  (define ones-target (ones n 1 #:device device))
  (define zeros-target (zeros n 1 #:device device))"""
assert old in s; s=s.replace(old,new)
old2="""GPU as the twin draws it. Images arrive from the loader in @tt{[0, 1]}
and are rescaled to the generator's range."""
new2="""GPU as the twin draws it. The two loss targets are built on
@racket[device] like the latent, so a caller may step a model that is not
on the default device. Images arrive from the loader in @tt{[0, 1]} and
are rescaled to the generator's range."""
assert old2 in s; s=s.replace(old2,new2)
open(p,'w').write(s)

# ---- H and I: ppm.rkt
p='torch/vision/ppm.rkt'
s=open(p).read()
old="""(define image/c
  (flat-named-contract
   'image
   (lambda (x)
     (and (tensor? x)
          (let ([dims (tensor-shape x)])
            (and (= 3 (length dims)) (= 3 (car dims))))))))"""
new=""";; ATen has no subtraction on a boolean tensor, so the range transform
;; has nothing to apply there
(define image/c
  (flat-named-contract
   'image
   (lambda (x)
     (and (tensor? x)
          (not (eq? (tensor-dtype x) 'bool))
          (let ([dims (tensor-shape x)])
            (and (= 3 (length dims)) (= 3 (car dims))))))))"""
assert old in s; s=s.replace(old,new)

old="""  (define cols (min columns n))
  (define rows (quotient (+ n cols -1) cols))
  (define grid"""
new="""  (cond
    [(= n 1) (one-image (select images 0 0) c h w)]
    [else (grid-of images n c h w columns padding pad-value)]))

;; make_grid returns a single image as it is, with no border
(define (one-image image c h w)
  (cond
    [(= c 1)
     (define out (full 0.0 3 h w
                       #:device (tensor-device image)
                       #:dtype (tensor-dtype image)))
     (copy! out image)
     out]
    [else image]))

(define (grid-of images n c h w columns padding pad-value)
  (define cols (min columns n))
  (define rows (quotient (+ n cols -1) cols))
  (define grid"""
assert old in s; s=s.replace(old,new)
open(p,'w').write(s)

# ---- H: the python twin gains the singleton case
p='torch/tests/python/make_grid_parity.py'
s=open(p).read()
old="""        nmaps = tensor.size(0)"""
new="""        if tensor.size(0) == 1:
            return tensor.squeeze(0)
        nmaps = tensor.size(0)"""
assert old in s; s=s.replace(old,new)
old="""print(json.dumps({
    "shape": list(g.shape),
    "values": [float(v) for v in g.flatten().tolist()],
    "pixels": [int(v) for v in q.contiguous().flatten().tolist()],
}))"""
new="""torch.manual_seed(1)
one = make_grid(torch.rand(1, 3, 4, 4), nrow=2, padding=1, pad_value=0.5)
print(json.dumps({
    "shape": list(g.shape),
    "values": [float(v) for v in g.flatten().tolist()],
    "pixels": [int(v) for v in q.contiguous().flatten().tolist()],
    "one_shape": list(one.shape),
    "one_values": [float(v) for v in one.flatten().tolist()],
}))"""
assert old in s; s=s.replace(old,new)
open(p,'w').write(s)

# ---- H and J: the cross test
p='torch/tests/python-cross-test.rkt'
s=open(p).read()
old="""       (check-equal? (bytes->list (subbytes bs (bytes-length header)))
                     (hash-ref j 'pixels)
                     "write-ppm: save_image's quantization"))"""
new="""       (define pixels (bytes->list (subbytes bs (bytes-length header))))
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
assert old in s; s=s.replace(old,new)
open(p,'w').write(s)
print("171 patch applied")
