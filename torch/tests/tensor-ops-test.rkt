#lang racket/base

;; Unit tests for the v1 op tranche (creation, shape, elementwise, reductions,
;; linalg, marshalling). Value parity with PyTorch lives in
;; generated-parity-test + python-cross-test; these pin the Racket-facing
;; behavior.

(module+ test
  (require (only-in racket/list drop take)
           rackunit
           "../main.rkt")

  (test-case "creation goldens"
    (check-equal? (tensor->list (zeros 2 2)) '(0.0 0.0 0.0 0.0))
    (check-equal? (tensor->list (ones 3)) '(1.0 1.0 1.0))
    (check-equal? (tensor->list (full 7.5 2)) '(7.5 7.5))
    (check-equal? (tensor->list (arange 3)) '(0.0 1.0 2.0))
    (check-equal? (tensor->list (arange 1 2.5 0.5)) '(1.0 1.5 2.0))
    (check-equal? (tensor->list (eye 2)) '(1.0 0.0 0.0 1.0))
    (check-equal? (tensor-shape (eye 2 3)) '(2 3)))

  (test-case "dtype inference mirrors torch.tensor (#44)"
    ;; the headline: all-integer literals are int64, printed bare
    (check-equal? (tensor-dtype (tensor '(1 2 3))) 'int64)
    (check-equal? (tensor->repr (tensor '(1 2 3))) "tensor([1, 2, 3])")
    ;; any inexact demotes the whole tensor to float32
    (check-equal? (tensor-dtype (tensor '(1 2.0 3))) 'float32)
    (check-equal? (tensor-dtype (tensor '(1.0 2.0))) 'float32)
    ;; scalars follow the same rule
    (check-equal? (tensor-dtype (tensor 5)) 'int64)
    (check-equal? (tensor-dtype (tensor 5.0)) 'float32)
    ;; int64 data round-trips exactly — (2^53)+1 is unrepresentable as a
    ;; double, so any float transit would corrupt it
    (check-equal? (tensor->list (tensor (list (+ (expt 2 53) 1))))
                  (list (+ (expt 2 53) 1)))
    ;; #:dtype overrides inference both ways; 'int64 truncates toward
    ;; zero (torch's cast semantics)
    (check-equal? (tensor-dtype (tensor '(1 2) #:dtype 'float32)) 'float32)
    ;; float64 marshals at double precision: 2^24+1 survives (the f32
    ;; path truncates it to 16777216)
    (check-equal? (tensor->list (to-dtype (tensor '(16777217)) 'float64))
                  '(16777217.0))
    (check-regexp-match #rx"16777217"
                        (tensor->repr
                         (to-dtype (tensor '(16777217)) 'float64)))
    (check-equal? (tensor->list (tensor '(1.9 -1.9) #:dtype 'int64)) '(1 -1))
    (check-exn exn:fail? (lambda () (tensor '(1 2) #:dtype 'float64)))
    ;; nested integer data flows through shape ops as int64
    (check-equal? (tensor->list (transpose (tensor '((1 2) (3 4))) 0 1))
                  '(1 3 2 4))
    ;; empty data stays float32, exactly torch.tensor([])
    (check-equal? (tensor-dtype (tensor '())) 'float32)
    (check-equal? (tensor->repr (tensor '())) "tensor([])")
    (check-equal? (tensor->repr (tensor '() #:dtype 'int64))
                  "tensor([], dtype=torch.int64)")
    ;; multi-dim empties carry torch's size= clause
    (check-equal? (tensor->repr (tensor '(() ())))
                  "tensor([], size=(2, 0))")
    (check-equal? (tensor->repr (tensor '(() ()) #:dtype 'int64))
                  "tensor([], size=(2, 0), dtype=torch.int64)")
    ;; the unprefixed property names alias the tensor- forms, and
    ;; `device` doubles as query (tensor arg) and constructor (symbol +
    ;; optional ordinal, default 0 — torch.device semantics)
    (let ([q (tensor '((1 2 3)) #:device (device 'cpu))])
      (check-equal? (shape q) '(1 3))
      (check-equal? (dtype q) 'int64)
      (check-equal? (numel q) 3)
      (check-equal? (device q) (cpu-device))
      (check-equal? (device 'cuda) (cuda-device 0))
      (check-equal? (device 'cuda 1) (cuda-device 1))
      ;; boundary rejections: an ordinal is only meaningful with 'cuda
      (check-exn exn:fail:contract? (lambda () (device q 1)))
      (check-exn exn:fail:contract? (lambda () (device 'cpu 1)))
      ;; ...but the one valid CPU ordinal stays accepted
      (check-equal? (device 'cpu 0) (cpu-device)))
    ;; comparisons produce genuine bool tensors, the query says so, and
    ;; the repr prints True/False like torch
    (check-equal? (dtype (eq (tensor '(1 2)) 1)) 'bool)
    (check-equal? (tensor->repr (eq (tensor '(1 2)) 1))
                  "tensor([ True, False])")
    ;; non-finite values have no int64 representation — tensor's own
    ;; error shape, as in torch
    (check-exn #rx"non-finite"
               (lambda () (tensor '(+inf.0) #:dtype 'int64)))
    ;; exact int64 comparisons never transit a double: 2^53 < 2^53+1
    ;; must hold (the scalar path would round the rhs down and flip it)
    (check-equal? (tensor->list (lt (tensor (expt 2 53))
                                    (+ (expt 2 53) 1)))
                  '(1.0))
    (check-equal? (tensor->list (eq (tensor (+ (expt 2 53) 1))
                                    (expt 2 53)))
                  '(0.0)))

  (test-case "large tensors summarize like PyTorch (#45)"
    ;; the headline hang: a 2^20-element repr returns instantly with the
    ;; edgeitems form (this test formerly hung the suite)
    (define r (tensor->repr (zeros 1024 1024)))
    (check-regexp-match #rx"\\.\\.\\." r)
    (check-true (< (string-length r) 400))
    (check-equal?
     (tensor->repr (tensor (build-list 2000 values)))
     "tensor([   0,    1,    2,  ..., 1997, 1998, 1999])")
    ;; strict threshold: 1000 elements print in full, 1001 summarize
    (check-false (regexp-match? #rx"\\.\\.\\." (tensor->repr (zeros 1000))))
    (check-regexp-match #rx"\\.\\.\\." (tensor->repr (zeros 1001)))
    ;; wide-dynamic-range large floats summarize in sci notation
    ;; (formerly the ATen fallback — which never summarized and would
    ;; have resurrected the hang)
    (manual-seed! 0)
    (let ([r (tensor->repr (mul (randn 2000) 1e10))])
      (check-regexp-match #rx"e[+][0-9][0-9]" r)
      (check-true (< (string-length r) 400)))
    ;; a dimension of exactly 2*edgeitems never elides, even inside a
    ;; summarized tensor: (zeros 6 200) summarizes (1200 elements), the
    ;; 200-wide rows elide inline, but all 6 rows print (no "...," row)
    (let ([r (tensor->repr (zeros 6 200))])
      (check-regexp-match #rx"\\.\\.\\." r)
      (check-false (regexp-match? #rx"\n *\\.\\.\\.," r))
      (check-equal? (length (regexp-match* #rx"\n" r)) 5)))

  (test-case "tensor from nested lists infers the shape"
    (define t (tensor '((1 2 3) (4 5 6))))
    (check-equal? (tensor-shape t) '(2 3))
    ;; all-integer literals infer int64 (#44): exact integers out
    (check-equal? (tensor->list t) '(1 2 3 4 5 6))
    (check-equal? (tensor-shape (tensor 5)) '())
    (check-equal? (tensor-shape (tensor '(1 2))) '(2))
    (check-exn exn:fail? (lambda () (tensor '((1 2) (3))))))

  (test-case "shape ops"
    (define t (tensor '((1 2 3) (4 5 6))))
    (check-equal? (tensor-shape (reshape t 3 2)) '(3 2))
    (check-equal? (tensor-shape (reshape t -1)) '(6))
    (check-equal? (tensor-shape (view t 6)) '(6))
    (check-equal? (tensor->list (transpose t 0 1)) '(1 4 2 5 3 6))
    (check-equal? (tensor-shape (permute t 1 0)) '(3 2))
    (check-equal? (tensor-shape (unsqueeze t 0)) '(1 2 3))
    (check-equal? (tensor-shape (squeeze (unsqueeze t 0))) '(2 3))
    (check-equal? (tensor-shape (squeeze (unsqueeze t 0) 0)) '(2 3))
    (check-equal? (tensor-shape (cat (list t t))) '(4 3))
    (check-equal? (tensor-shape (cat (list t t) 1)) '(2 6))
    (check-equal? (tensor-shape (stack (list t t))) '(2 2 3)))

  (test-case "elementwise with scalar dispatch on either side"
    (define x (tensor '(1 -2 3)))
    ;; tensor⊕tensor stays int64 — torch's integer arithmetic (#44)
    (check-equal? (tensor->list (add x x)) '(2 -4 6))
    ;; KNOWN DEVIATION: scalar operands marshal as C doubles, so an
    ;; int64 tensor ⊕ exact-integer scalar promotes to float32 — PyTorch
    ;; keeps int64 for int scalars. Separable scalar-promotion gap;
    ;; tracked as #44 follow-up.
    (check-equal? (tensor->list (add x 1)) '(2.0 -1.0 4.0))
    (check-equal? (tensor->list (add 1 x)) '(2.0 -1.0 4.0))
    (check-equal? (tensor->list (sub x 1)) '(0.0 -3.0 2.0))
    (check-equal? (tensor->list (sub 1 x)) '(0.0 3.0 -2.0))
    (check-equal? (tensor->list (mul x 2)) '(2.0 -4.0 6.0))
    (check-equal? (tensor->list (div x 2)) '(0.5 -1.0 1.5))
    (check-equal? (tensor->list (div 6 (tensor '(1 2 3)))) '(6.0 3.0 2.0))
    ;; unary ops stay int64
    (check-equal? (tensor->list (neg x)) '(-1 2 -3))
    (check-equal? (tensor->list (relu x)) '(1 0 3))
    (check-equal? (tensor->list (pow x 2)) '(1.0 4.0 9.0)))

  (test-case "gelu (exact erf form): x * Phi(x)"
    ;; float literals: gelu is float-only in torch (as in Python)
    (define g (gelu (tensor '(0.0 1.0 -1.0))))
    (check-= (car (tensor->list g)) 0.0 1e-6)
    (check-= (cadr (tensor->list g)) 0.841345 1e-5)
    (check-= (caddr (tensor->list g)) -0.158655 1e-5))

  (test-case "tril/triu default diagonal + offsets"
    (define m (reshape (arange 1 10) 3 3))
    (check-equal? (tensor->list (tril m))
                  '(1.0 0.0 0.0 4.0 5.0 0.0 7.0 8.0 9.0))
    (check-equal? (tensor->list (triu m))
                  '(1.0 2.0 3.0 0.0 5.0 6.0 0.0 0.0 9.0))
    (check-equal? (tensor->list (tril m -1))
                  '(0.0 0.0 0.0 4.0 0.0 0.0 7.0 8.0 0.0))
    (check-equal? (tensor->list (triu m 1))
                  '(0.0 2.0 3.0 0.0 0.0 6.0 0.0 0.0 0.0)))

  (test-case "masked-fill via a comparison-built bool mask"
    (define x (tensor '(10 20 30 40)))
    (define mask (ne (tensor '(0 1 0 1)) 0))
    (check-equal? (tensor->list (masked-fill x mask -100))
                  '(10 -100 30 -100))
    ;; the causal-mask composition: upper triangle goes to the fill value.
    (define causal (eq (tril (ones 2 2)) 0))
    (check-equal? (tensor->list (masked-fill (ones 2 2) causal 0))
                  '(1.0 0.0 1.0 1.0))
    ;; a [T,T] mask broadcasts over batched [B,T,T] scores — the attention
    ;; shape the training loop uses; each batch slice gets the same mask.
    (check-equal? (tensor->list (masked-fill (ones 2 2 2) causal 0))
                  '(1.0 0.0 1.0 1.0 1.0 0.0 1.0 1.0)))

  (test-case "embedding #:padding-idx marshals a real index"
    ;; forward output is padding-idx-insensitive in ATen (it gates the
    ;; backward), so the check is that a non-#f index marshals and gathers
    ;; identically — the #f->-1 default mapping isn't the only tested path.
    (define weight (reshape (arange 1 9) 4 2))
    (define indices (to-dtype (tensor '(2 0 2)) 'int64))
    (check-equal? (tensor->list (embedding indices weight #:padding-idx 0))
                  (tensor->list (embedding indices weight))))

  (test-case "embedding gathers rows, F.embedding arg order"
    (define weight (reshape (arange 1 9) 4 2))
    (define indices (to-dtype (tensor '(2 0 2)) 'int64))
    (define out (embedding indices weight))
    (check-equal? (tensor-shape out) '(3 2))
    (check-equal? (tensor->list out) '(5.0 6.0 1.0 2.0 5.0 6.0)))

  (test-case "layer-norm: bare + affine, int normalized-shape"
    ;; float literals: layer-norm is float-only in torch (as in Python)
    (define x (tensor '((1.0 2.0 3.0) (4.0 6.0 8.0))))
    (define bare (layer-norm x 3))
    (check-equal? (tensor-shape bare) '(2 3))
    (check-= (car (tensor->list bare)) -1.2247 1e-4)
    (check-= (cadr (tensor->list bare)) 0.0 1e-4)
    (check-= (caddr (tensor->list bare)) 1.2247 1e-4)
    ;; affine: y = normalized * weight + bias.
    (define affine
      (layer-norm x '(3) #:weight (mul (ones 3) 2) #:bias (ones 3)))
    (for ([a (in-list (tensor->list affine))]
          [b (in-list (tensor->list bare))])
      (check-= a (+ (* 2 b) 1) 1e-4))
    ;; multi-dim normalized-shape: stats pool over the trailing [2,3] dims
    ;; jointly, so each [2,3] slice comes out zero-mean — distinct from
    ;; per-row normalization, which would zero each row separately.
    (define m (tensor '(((1.0 2.0 3.0) (4.0 6.0 8.0))
                        ((10.0 20.0 30.0) (40.0 60.0 80.0)))))
    (define nm (layer-norm m '(2 3)))
    (check-equal? (tensor-shape nm) '(2 2 3))
    (define vals (tensor->list nm))
    (check-= (apply + (take vals 6)) 0.0 1e-3)
    (check-= (apply + (drop vals 6)) 0.0 1e-3)
    ;; rows within a slice keep distinct means (joint stats, not per-row).
    (check-true (> (abs (- (apply + (take vals 3))
                           (apply + (take (drop vals 3) 3))))
                   0.1)))

  (test-case "exp/log/sqrt/tanh/max/min fall back to racket/base on numbers"
    (check-equal? (exp 0) 1)
    (check-equal? (log 1) 0)
    (check-equal? (log 8 2) 3.0)
    (check-equal? (sqrt 4) 2)
    (check-equal? (tanh 0) 0)
    (check-= (tanh 0.5) 0.46211715726 1e-9)
    (check-equal? (max 1 2 3) 3)
    (check-equal? (min 1 2 3) 1)
    (check-= (item (tanh (tensor '(0.5)))) 0.4621171 1e-5)
    (define x (tensor '(1 4 9)))
    (check-equal? (tensor->list (sqrt x)) '(1.0 2.0 3.0))
    (check-= (item (exp (tensor 0))) 1.0 1e-6)
    (check-= (item (log (tensor 1))) 0.0 1e-6)
    (check-equal? (item (max x)) 9)
    (check-equal? (item (min x)) 1))

  (test-case "reductions"
    ;; float literals: torch (ours and Python's) rejects mean on int64
    (define t (tensor '((1.0 2.0) (3.0 4.0))))
    (check-equal? (item (sum t)) 10.0)
    (check-equal? (item (mean t)) 2.5)
    ;; argmax returns int64 indices — exact integers out (#44), and
    ;; item on an int64 scalar is exact too
    (check-equal? (tensor->list (argmax t 1)) '(1 1))
    (check-equal? (item (argmax t)) 3)
    (check-equal? (tensor-shape (argmax t 1 #:keepdim #t)) '(2 1))
    (define p (softmax t 1))
    (define vals (tensor->list p))
    (check-= (+ (car vals) (cadr vals)) 1.0 1e-6)
    (check-= (item (sum (exp (log-softmax t 1)))) 2.0 1e-5))

  (test-case "linalg"
    (define a (tensor '((1 2) (3 4))))
    (define v (tensor '(1 1)))
    (check-equal? (tensor->list (matmul a a)) '(7 10 15 22))
    (check-equal? (tensor->list (mm a a)) '(7 10 15 22))
    (check-equal? (tensor->list (mv a v)) '(3 7))
    ;; int64 dot -> exact int64 item (#44)
    (check-equal? (item (dot v v)) 2)
    ;; shape mismatch surfaces as a Racket error carrying the C++ message
    (check-exn exn:fail? (lambda () (mv a (tensor '(1 1 1))))))

  (test-case "item and to-dtype"
    ;; int64 scalars item out EXACT (#44) — including past 2^53, where
    ;; the double path would round
    (check-equal? (item (tensor 42)) 42)
    (check-equal? (item (tensor 42.0)) 42.0)
    (check-equal? (item (tensor (+ (expt 2 53) 1))) (+ (expt 2 53) 1))
    (check-exn exn:fail? (lambda () (item (tensor '(1 2)))))
    (define i (to-dtype (tensor '(1.5 2.5)) 'int64))
    ;; int64 marshals out exact (#44)
    (check-equal? (tensor->list i) '(1 2)))

  (test-case "+ - * / dispatch on tensors and stay racket/base on numbers"
    (check-equal? (+ 1 2 3) 6)
    (check-equal? (* 2 3 4) 24)
    (check-equal? (- 5 1) 4)
    (check-equal? (/ 6 3) 2)
    (check-equal? (+) 0)
    (check-equal? (*) 1)
    (define x (tensor '(1 2)))
    ;; scalar forms promote to float (the documented scalar-marshal
    ;; deviation above); unary neg and tensor×tensor stay int64
    (check-equal? (tensor->list (+ x 1)) '(2.0 3.0))
    (check-equal? (tensor->list (+ 1 x 1)) '(3.0 4.0))
    (check-equal? (tensor->list (- x)) '(-1 -2))
    (check-equal? (tensor->list (- 3 x)) '(2.0 1.0))
    (check-equal? (tensor->list (* x (tensor '(3 4)))) '(3 8))
    ;; division promotes to float (torch true-division), even on int64
    (check-equal? (tensor->list (/ x)) '(1.0 0.5))
    (check-equal? (tensor->list (/ x 2)) '(0.5 1.0)))

  (test-case "@ is matmul and chains left like Python"
    ;; float literals: eye is float32, and torch rejects mixed
    ;; int64 @ float32 matmul (Python errors identically)
    (define a (tensor '((1.0 2.0) (3.0 4.0))))
    (define i (eye 2))
    (check-equal? (tensor->list (@ a i)) '(1.0 2.0 3.0 4.0))
    (check-equal? (tensor->list (@ a i a)) (tensor->list (matmul a a))))

  (test-case "t and Σ aliases, threading pipelines"
    (define a (tensor '((1 2) (3 4))))
    (check-equal? (tensor->list (t a 0 1)) (tensor->list (transpose a 0 1)))
    (check-equal? (item (Σ a)) 10)
    (define x (tensor '(1 2 3)))
    (check-equal? (item (~> x (* x) Σ)) 14))

  (test-case "tensor #:requires-grad? marks the leaf at construction"
    ;; float literals: integer literals infer int64 (#44), and torch
    ;; rejects requires-grad on integer tensors (as does Python)
    (define x (tensor '(1.0 2.0) #:requires-grad? #t))
    (check-true (requires-grad? x))
    (check-false (requires-grad? (tensor '(1.0 2.0))))
    ;; ...and the int64 rejection is itself parity behavior
    (check-exn exn:fail?
               (lambda () (tensor '(1 2) #:requires-grad? #t))))

  (test-case "wrong call shapes get contract blame at the facade"
    (check-exn exn:fail:contract? (lambda () (add 1 2)))
    (check-exn exn:fail:contract? (lambda () (sub 1.0 2.0)))
    (check-exn exn:fail:contract? (lambda () (log (tensor '(1 2)) 2)))
    (check-exn exn:fail:contract? (lambda () (max (tensor '(1 2)) 3)))
    (check-exn exn:fail:contract? (lambda () (argmax even?)))
    (check-exn exn:fail:contract?
               (lambda () (argmax even? '(1 2) #:keepdim #t))))

  (test-case "rand and uniform! stay in range"
    (manual-seed! 0)
    (define u (rand 64))
    (for ([x (in-list (tensor->list u))])
      (check-true (and (>= x 0.0) (< x 1.0))))
    (define w (zeros 64))
    (uniform! w -2.0 -1.0)
    (for ([x (in-list (tensor->list w))])
      (check-true (and (>= x -2.0) (< x -1.0)))))

  (test-case "comparison dispatchers: tensor rhs and real rhs agree"
    (define a (tensor '(1 2 3)))
    ;; tensor-vs-tensor branch
    (check-equal? (tensor->list (eq a (tensor '(1 5 3)))) '(1.0 0.0 1.0))
    (check-equal? (tensor->list (lt a (tensor '(2 2 2)))) '(1.0 0.0 0.0))
    ;; tensor-vs-real branch (exact->inexact path) -- same masks
    (check-equal? (tensor->list (ge a 2)) '(0.0 1.0 1.0))
    (check-equal? (tensor->list (ne a 2)) '(1.0 0.0 1.0))
    (check-equal? (tensor->list (gt a 2.5)) '(0.0 0.0 1.0)))

  (test-case "flatten collapses dims and rejects an invalid range"
    ;; 4-d so end-dim=1 and end-dim=-1 give distinct shapes.
    (define t (reshape (arange 120) 2 3 4 5))
    (check-equal? (tensor-shape (flatten t)) '(120))
    (check-equal? (tensor-shape (flatten t 1)) '(2 60))      ; end -1
    (check-equal? (tensor-shape (flatten t 1 2)) '(2 12 5))  ; end 2
    ;; an explicit negative end-dim passes the facade contract (index/c is
    ;; exact-integer?, not nonnegative) and normalizes like PyTorch.
    (check-equal? (tensor-shape (flatten t 0 -1)) '(120))
    ;; 0-d tensor flattens to shape (1), matching PyTorch.
    (check-equal? (tensor-shape (flatten (zeros))) '(1))
    ;; a non-tensor defers to racket/list's flatten.
    (check-equal? (flatten '(1 (2 3))) '(1 2 3))
    ;; start > end after normalization is rejected, not silently mis-sliced.
    (check-exn #rx"invalid dim range" (lambda () (flatten t 3 1))))

  (test-case "narrow returns a view aliasing the source storage"
    ;; float literals: uniform! below is a float-only op (as in Python)
    (define t (tensor '(1.0 2.0 3.0 4.0)))
    (define v (narrow t 0 1 2))
    (check-equal? (tensor->list v) '(2.0 3.0))
    ;; an in-place write through the view mutates the original (torch.narrow
    ;; semantics), proving the result aliases rather than copies.
    (uniform! v 0.0 0.0)
    (check-equal? (tensor->list t) '(1.0 0.0 0.0 4.0))))
