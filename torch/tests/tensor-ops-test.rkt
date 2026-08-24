#lang racket/base

(module+ test
  (require (only-in ffi/vector f32vector s64vector)
           (only-in racket/list drop)
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
    (check-equal? (tensor-dtype (tensor '(1 2 3))) 'int64)
    (check-equal? (tensor->repr (tensor '(1 2 3))) "tensor([1, 2, 3])")
    (check-equal? (tensor-dtype (tensor '(1 2.0 3))) 'float32)
    (check-equal? (tensor-dtype (tensor '(1.0 2.0))) 'float32)
    (check-equal? (tensor-dtype (tensor 5)) 'int64)
    (check-equal? (tensor-dtype (tensor 5.0)) 'float32)
    ;; 2^53+1 is unrepresentable as a double — proves no float transit
    (check-equal? (tensor->list (tensor (list (+ (expt 2 53) 1))))
                  (list (+ (expt 2 53) 1)))
    (check-equal? (tensor-dtype (tensor '(1 2) #:dtype 'float32)) 'float32)
    ;; 2^24+1 survives the f64 marshal (the f32 path truncates to 16777216)
    (check-equal? (tensor->list (to-dtype (tensor '(16777217)) 'float64))
                  '(16777217.0))
    (check-regexp-match #rx"16777217"
                        (tensor->repr
                         (to-dtype (tensor '(16777217)) 'float64)))
    (check-equal? (tensor->list (tensor '(1.9 -1.9) #:dtype 'int64)) '(1 -1))
    (check-exn exn:fail? (lambda () (tensor '(1 2) #:dtype 'float64)))
    (check-equal? (tensor->list (transpose (tensor '((1 2) (3 4))) 0 1))
                  '(1 3 2 4))
    (check-equal? (tensor-dtype (tensor '())) 'float32)
    (check-equal? (tensor->repr (tensor '())) "tensor([])")
    (check-equal? (tensor->repr (tensor '() #:dtype 'int64))
                  "tensor([], dtype=torch.int64)")
    (check-equal? (tensor->repr (tensor '(() ())))
                  "tensor([], size=(2, 0))")
    (check-equal? (tensor->repr (tensor '(() ()) #:dtype 'int64))
                  "tensor([], size=(2, 0), dtype=torch.int64)")
    (let ([q (tensor '((1 2 3)) #:device (device 'cpu))])
      (check-equal? (shape q) '(1 3))
      (check-equal? (dtype q) 'int64)
      (check-equal? (numel q) 3)
      (check-equal? (device q) (cpu-device))
      (check-equal? (device 'cuda) (cuda-device 0))
      (check-equal? (device 'cuda 1) (cuda-device 1))
      (check-exn exn:fail:contract? (lambda () (device q 1)))
      (check-exn exn:fail:contract? (lambda () (device 'cpu 1)))
      (check-equal? (device 'cpu 0) (cpu-device)))
    (check-equal? (dtype (eq (tensor '(1 2)) 1)) 'bool)
    (check-equal? (tensor->repr (eq (tensor '(1 2)) 1))
                  "tensor([ True, False])")
    (check-exn #rx"non-finite"
               (lambda () (tensor '(+inf.0) #:dtype 'int64)))
    ;; 2^53 < 2^53+1 must hold exactly — a double transit would flip it
    (check-equal? (tensor->list (lt (tensor (expt 2 53))
                                    (+ (expt 2 53) 1)))
                  '(1.0))
    (check-equal? (tensor->list (eq (tensor (+ (expt 2 53) 1))
                                    (expt 2 53)))
                  '(0.0)))

  (test-case "large tensors summarize like PyTorch (#45)"
    (define r (tensor->repr (zeros 1024 1024)))
    (check-regexp-match #rx"\\.\\.\\." r)
    (check-true (< (string-length r) 400))
    (check-equal?
     (tensor->repr (tensor (build-list 2000 values)))
     "tensor([   0,    1,    2,  ..., 1997, 1998, 1999])")
    (check-false (regexp-match? #rx"\\.\\.\\." (tensor->repr (zeros 1000))))
    (check-regexp-match #rx"\\.\\.\\." (tensor->repr (zeros 1001)))
    (manual-seed! 0)
    (let ([r (tensor->repr (mul (randn 2000) 1e10))])
      (check-regexp-match #rx"e[+][0-9][0-9]" r)
      (check-true (< (string-length r) 400)))
    ;; a dimension of exactly 2*edgeitems (6 rows) never elides
    (let ([r (tensor->repr (zeros 6 200))])
      (check-regexp-match #rx"\\.\\.\\." r)
      (check-false (regexp-match? #rx"\n *\\.\\.\\.," r))
      (check-equal? (length (regexp-match* #rx"\n" r)) 5)))

  (test-case "tensor from nested lists infers the shape"
    (define t (tensor '((1 2 3) (4 5 6))))
    (check-equal? (tensor-shape t) '(2 3))
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
    (check-equal? (tensor->list (add x x)) '(2 -4 6))
    ;; KNOWN DEVIATION (#44 follow-up): scalars marshal as C doubles, so
    ;; int64 tensor ⊕ integer scalar promotes to float32; PyTorch keeps int64.
    (check-equal? (tensor->list (add x 1)) '(2.0 -1.0 4.0))
    (check-equal? (tensor->list (add 1 x)) '(2.0 -1.0 4.0))
    (check-equal? (tensor->list (sub x 1)) '(0.0 -3.0 2.0))
    (check-equal? (tensor->list (sub 1 x)) '(0.0 3.0 -2.0))
    (check-equal? (tensor->list (mul x 2)) '(2.0 -4.0 6.0))
    (check-equal? (tensor->list (div x 2)) '(0.5 -1.0 1.5))
    (check-equal? (tensor->list (div 6 (tensor '(1 2 3)))) '(6.0 3.0 2.0))
    (check-equal? (tensor->list (neg x)) '(-1 2 -3))
    (check-equal? (tensor->list (relu x)) '(1 0 3))
    (check-equal? (tensor->list (pow x 2)) '(1.0 4.0 9.0)))

  (test-case "gelu (exact erf form): x * Phi(x)"
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

  (test-case "masked-fill: comparison-built bool mask, [T,T] over [B,T,T]"
    (define x (tensor '(10 20 30 40)))
    (define mask (ne (tensor '(0 1 0 1)) 0))
    (check-equal? (tensor->list (masked-fill x mask -100))
                  '(10 -100 30 -100))
    (define causal (eq (tril (ones 2 2)) 0))
    (check-equal? (tensor->list (masked-fill (ones 2 2) causal 0))
                  '(1.0 0.0 1.0 1.0))
    (check-equal? (tensor->list (masked-fill (ones 2 2 2) causal 0))
                  '(1.0 0.0 1.0 1.0 1.0 0.0 1.0 1.0)))

  (test-case "embedding #:padding-idx marshals; ATen's forward ignores it"
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
    (define x (tensor '((1.0 2.0 3.0) (4.0 6.0 8.0))))
    (define bare (layer-norm x 3))
    (check-equal? (tensor-shape bare) '(2 3))
    (check-= (car (tensor->list bare)) -1.2247 1e-4)
    (check-= (cadr (tensor->list bare)) 0.0 1e-4)
    (check-= (caddr (tensor->list bare)) 1.2247 1e-4)
    (define affine
      (layer-norm x '(3) #:weight (mul (ones 3) 2) #:bias (ones 3)))
    (for ([a (in-list (tensor->list affine))]
          [b (in-list (tensor->list bare))])
      (check-= a (+ (* 2 b) 1) 1e-4))
    ;; multi-dim normalized-shape: joint stats over the trailing [2,3]
    ;; dims, not per-row
    (define m (tensor '(((1.0 2.0 3.0) (4.0 6.0 8.0))
                        ((10.0 20.0 30.0) (40.0 60.0 80.0)))))
    (define nm (layer-norm m '(2 3)))
    (check-equal? (tensor-shape nm) '(2 2 3))
    (define vals (tensor->list nm))
    (check-= (apply + (take vals 6)) 0.0 1e-3)
    (check-= (apply + (drop vals 6)) 0.0 1e-3)
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
    (define t (tensor '((1.0 2.0) (3.0 4.0))))
    (check-equal? (item (sum t)) 10.0)
    (check-equal? (item (mean t)) 2.5)
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
    (check-equal? (item (dot v v)) 2)
    (check-exn exn:fail? (lambda () (mv a (tensor '(1 1 1))))))

  (test-case "item and to-dtype"
    ;; 2^53+1 items out exact — the double path would round it
    (check-equal? (item (tensor 42)) 42)
    (check-equal? (item (tensor 42.0)) 42.0)
    (check-equal? (item (tensor (+ (expt 2 53) 1))) (+ (expt 2 53) 1))
    (check-exn exn:fail? (lambda () (item (tensor '(1 2)))))
    (define i (to-dtype (tensor '(1.5 2.5)) 'int64))
    (check-equal? (tensor->list i) '(1 2)))

  (test-case "+ - * / dispatch on tensors and stay racket/base on numbers"
    (check-equal? (+ 1 2 3) 6)
    (check-equal? (* 2 3 4) 24)
    (check-equal? (- 5 1) 4)
    (check-equal? (/ 6 3) 2)
    (check-equal? (+) 0)
    (check-equal? (*) 1)
    (define x (tensor '(1 2)))
    (check-equal? (tensor->list (+ x 1)) '(2.0 3.0))
    (check-equal? (tensor->list (+ 1 x 1)) '(3.0 4.0))
    (check-equal? (tensor->list (- x)) '(-1 -2))
    (check-equal? (tensor->list (- 3 x)) '(2.0 1.0))
    (check-equal? (tensor->list (* x (tensor '(3 4)))) '(3 8))
    (check-equal? (tensor->list (/ x)) '(1.0 0.5))
    (check-equal? (tensor->list (/ x 2)) '(0.5 1.0)))

  (test-case "@ is matmul and chains left like Python"
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
    (define x (tensor '(1.0 2.0) #:requires-grad? #t))
    (check-true (requires-grad? x))
    (check-false (requires-grad? (tensor '(1.0 2.0))))
    (check-exn exn:fail?
               (lambda () (tensor '(1 2) #:requires-grad? #t))))

  (test-case "sequence ingestion: vectors, math vectors, mixed nesting (#55)"
    (check-equal? (tensor->list (tensor (vector 1 2 3))) '(1 2 3))
    (check-equal? (dtype (tensor (vector 1 2 3))) 'int64)
    (check-equal? (dtype (tensor (vector 1 2.0))) 'float32)
    (check-equal? (shape (tensor (vector (vector 1.0 2.0) (vector 3.0 4.0))))
                  '(2 2))
    (check-equal? (tensor->list (tensor (list (vector 1 2) (list 3 4))))
                  '(1 2 3 4))
    (check-equal? (shape (tensor (vector '((1.0) (2.0)) '((3.0) (4.0)))))
                  '(2 2 1))
    (check-equal? (shape (tensor (list (f32vector 1.0 2.0)
                                       (f32vector 3.0 4.0))))
                  '(2 2))
    (check-equal? (dtype (tensor (list (s64vector 1 2)))) 'int64)
    (check-equal? (tensor->list (tensor (f32vector 1.5 2.5))) '(1.5 2.5))
    (check-equal? (dtype (tensor (f32vector 1.5))) 'float32)
    ;; zero-copy path: content past 2^53 proves no float transit
    (check-equal? (tensor->list (tensor (s64vector (add1 (expt 2 53)))))
                  (list (add1 (expt 2 53))))
    (check-equal? (dtype (tensor (s64vector 1))) 'int64)
    (check-equal? (tensor->list (tensor (f32vector 1.9) #:dtype 'int64))
                  '(1))
    (check-equal? (tensor->list (tensor (s64vector 3) #:dtype 'float32))
                  '(3.0))
    (check-equal? (dtype (tensor (vector))) 'float32)
    (check-equal? (tensor->repr (tensor (vector (vector) (vector))))
                  "tensor([], size=(2, 0))")
    (check-exn #rx"ragged"
               (lambda () (tensor (vector (vector 1 2) (vector 3)))))
    (check-exn #rx"ragged"
               (lambda () (tensor (list (f32vector 1.0 2.0)
                                        (f32vector 3.0)))))
    (check-exn #rx"ragged"
               (lambda ()
                 (tensor (vector (vector 1 2)
                                 (vector (vector 3) (vector 4))))))
    (check-exn #rx"ragged"
               (lambda ()
                 (tensor (list (list 1 2)
                               (vector (vector 3) (vector 4))))))
    (check-exn #rx"ragged"
               (lambda () (tensor (list (list 1) 2)))))

  (test-case "select views and tensor-ref sugar (#46)"
    (define t (tensor '((1 2 3) (4 5 6))))
    (check-equal? (tensor->list (select t 0 1)) '(4 5 6))
    (check-equal? (tensor->list (select t 1 -1)) '(3 6))
    (check-equal? (shape (select t 0 0)) '(3))
    (let ([m (tensor '((1.0 2.0) (3.0 4.0)))])
      (sub! (select m 0 0) (select m 0 0))
      (check-equal? (tensor->list m) '(0.0 0.0 3.0 4.0)))
    (check-equal? (ref t 1 2) 6)
    (check-equal? (tensor-ref t -1 -1) 6)
    (check-equal? (ref (tensor '((1.5 2.5))) 0 1) 2.5)
    (check-equal? (ref (tensor 7)) 7)
    ;; 2^53+1 survives because the walk stays device-resident until item
    (check-equal? (ref (tensor (list (add1 (expt 2 53)))) 0)
                  (add1 (expt 2 53)))
    (check-exn exn:fail? (lambda () (select t 0 5))))

  (test-case "ref mirrors python indexing: slices, ellipsis, None, masks (#46)"
    (define t (tensor '((1 2 3) (4 5 6))))
    (check-equal? (tensor->list (ref t 0)) '(1 2 3))
    (check-equal? (tensor->list (ref (arange 6) (:: 1 3))) '(1.0 2.0))
    (check-equal? (tensor->list (ref (arange 6) (:: #f #f 2))) '(0.0 2.0 4.0))
    (check-equal? (tensor->list (ref t (::) (:: 1 #f))) '(2 3 5 6))
    (check-equal? (shape (ref t (:: 1) (:: 2))) '(1 2))
    (let ([m (tensor '((1.0 2.0) (3.0 4.0)))])
      (sub! (ref m (:: 0 1)) (ref m (:: 0 1)))
      (check-equal? (tensor->list m) '(0.0 0.0 3.0 4.0)))
    (check-equal? (tensor->list (ref t '... 0)) '(1 4))
    (check-equal? (shape (ref t (::) #f)) '(2 1 3))
    (check-equal? (shape (ref t '... #f)) '(2 3 1))
    (check-equal? (tensor->list
                   (ref (tensor '(((1 2) (3 4)) ((5 6) (7 8)))) 1 '... 0))
                  '(5 7))
    (check-equal? (tensor->list (ref t (gt t 4))) '(5 6))
    ;; a rank-1 mask keeps python's dimension semantics: rows where true
    (check-equal? (tensor->list (ref t (ne (tensor '(0 1)) 0)))
                  '(4 5 6))
    (check-equal? (shape (ref t (ne (tensor '(0 1)) 0))) '(1 3))
    (check-equal? (tensor->list (ref t (::) (ne (tensor '(1 0 1)) 0)))
                  '(1 3 4 6))
    (check-equal? (shape (ref t '(0 0 1))) '(3 3))
    (check-equal? (tensor->list (ref t '(1 0) 0)) '(4 1))
    (check-equal? (tensor->list (ref t '(-1 0) 0)) '(4 1))
    (check-equal? (tensor->list (ref t '#(1 0) 0)) '(4 1))
    (check-equal? (tensor->list (ref t '#(-1 0) 0)) '(4 1))
    (check-equal? (tensor->list
                   (ref (arange 5) (to-dtype (tensor '(-1 0)) 'int64)))
                  '(4.0 0.0))
    (check-equal? (ref (gt t 4) 1 2) #t)
    (check-equal? (ref (gt t 4) 0 0) #f)
    (check-exn #rx"too many indices" (lambda () (ref t 0 0 0)))
    (check-exn #rx"at most one" (lambda () (ref t '... '...)))
    (let ([cube (tensor '(((1 2) (3 4)) ((5 6) (7 8))))])
      ;; a rank-m mask consumes m dims: python's masked-dims collapse
      (check-equal? (shape (ref cube (ne (tensor '((1 0) (0 1))) 0))) '(2 2))
      (check-equal? (tensor->list (ref cube (ne (tensor '((1 0) (0 1))) 0)))
                    '(1 2 7 8)))
    (check-exn #rx"too many indices" (lambda () (ref t (gt t 4) 0)))
    (check-exn #rx"mask shape" (lambda () (ref t 0 (ne (tensor '(1 0)) 0))))
    ;; python rejects broadcastable-but-unequal full-rank masks
    (check-exn #rx"mask shape"
               (lambda () (ref t (ne (tensor '((1 0 1))) 0))))
    (check-exn exn:fail:contract?
               (lambda () (ref t (tensor '(0.5 1.0)))))
    (check-exn exn:fail:contract? (lambda () (ref t '#(0.5))))
    (check-exn exn:fail:contract?
               (lambda () (ref t (to-dtype (tensor '((0 1))) 'int64))))
    (check-exn exn:fail:contract?
               (lambda () (ref t 0 (gt (tensor 1) 0))))
    (check-exn exn:fail:contract?
               (lambda () (tensor-ref t (tensor '(0.5 1.0)))))
    (check-exn exn:fail:contract?
               (lambda () (tensor-ref t 0 (gt (tensor 1) 0))))
    (check-exn exn:fail:contract?
               (lambda () (ref t (: "bad" 2))))
    (check-exn exn:fail:contract?
               (lambda () (ref t (: 0 2 1.5))))
    (check-exn exn:fail? (lambda () (ref (arange 6) (:: #f #f -1)))))

  (test-case "ref macro sugar expands to tensor-ref value specs (#46)"
    (define t (tensor '((1 2 3) (4 5 6))))
    (check-equal? (tensor->list (ref t : 0))
                  (tensor->list (tensor-ref t (::) 0)))
    (check-equal? (tensor->list (ref t (: 1 3)))
                  (tensor->list (tensor-ref t (:: 1 3))))
    (check-equal? (tensor->list (ref t (: 2)))
                  (tensor->list (tensor-ref t (:: 2))))
    (check-equal? (tensor->list (ref t (:~ 1)))
                  (tensor->list (tensor-ref t (:: 1 #f))))
    (check-equal? (tensor->list (ref t (: 1 _)))
                  (tensor->list (tensor-ref t (:: 1 #f))))
    (check-equal? (tensor->list (ref t : (: _ _ 2)))
                  (tensor->list (tensor-ref t (::) (:: #f #f 2))))
    (check-equal? (tensor->list (ref t : (:~ 0 2)))
                  (tensor->list (tensor-ref t (::) (:: 0 #f 2))))
    (check-equal? (tensor->list (ref t : (:~ _ 2)))
                  (tensor->list (tensor-ref t (::) (:: #f #f 2))))
    (check-equal? (tensor->list (ref t (:~ 1 _)))
                  (tensor->list (tensor-ref t (:: 1 #f))))
    (check-equal? (tensor->list (ref t : (: 0 _ 2)))
                  (tensor->list (tensor-ref t (::) (:: 0 #f 2))))
    (check-equal? (tensor->list (ref t (: 0 2 _)))
                  (tensor->list (tensor-ref t (:: 0 2))))
    (check-equal? (tensor->list (ref t .. 0))
                  (tensor->list (tensor-ref t '... 0)))
    (check-equal? (shape (ref t : _)) (shape (tensor-ref t (::) #f)))
    (check-equal? (shape (ref t _)) '(1 2 3))
    (let ([lo 0])
      (check-equal? (tensor->list (ref t (: (add1 lo) (+ 1 2)) 0))
                    '(4)))
    (let ([s (:: 1 3)])
      (check-equal? (tensor->list (ref t 0 s)) '(2 3)))
    (check-equal? (tensor->list (ref t (gt t 4))) '(5 6))
    (check-equal? (apply tensor-ref t (list 1 2)) 6)
    ;; `..` survives inside another macro's template (a literal `...`
    ;; there would be the macro ellipsis and fail to expand)
    (let-syntax ([first-of-last-dim
                  (syntax-rules () [(_ x) (ref x .. 0)])])
      (check-equal? (tensor->list (first-of-last-dim t)) '(1 4))))

  (test-case "selection family: take/gather/take-along-dim/where (#73)"
    (define t (reshape (arange 6) 2 3))
    (check-equal? (tensor->list (take t '(0 5 3))) '(0.0 5.0 3.0))
    (check-equal? (tensor->list (take t '#(5 0))) '(5.0 0.0))
    (check-equal? (take '(1 2 3) 2) '(1 2))
    (check-equal? (tensor->list
                   (gather t 1 (to-dtype (tensor '((0 2) (1 0))) 'int64)))
                  '(0.0 2.0 4.0 3.0))
    (check-equal? (tensor->list
                   (take-along-dim t (to-dtype (tensor '((0) (2))) 'int64) 1))
                  '(0.0 5.0))
    ;; dim omitted operates on the flattened tensor, like python
    (check-equal? (tensor->list
                   (take-along-dim t (to-dtype (tensor '(5 0)) 'int64)))
                  '(5.0 0.0))
    (check-equal? (map tensor->list (where (gt t 2.0)))
                  '((1 1 1) (0 1 2)))
    (check-equal? (tensor->list (where (gt t 2.0) t -1)) '(-1.0 -1.0 -1.0 3.0 4.0 5.0))
    (check-equal? (tensor->list (where (gt t 2.0) t (zeros 2 3)))
                  '(0.0 0.0 0.0 3.0 4.0 5.0))
    (check-equal? (tensor->list
                   (index-select t 1 (to-dtype (tensor '(2 0)) 'int64)))
                  '(2.0 0.0 5.0 3.0))
    (check-equal? (tensor->list (masked-select t (gt t 3.0))) '(4.0 5.0))
    (check-equal? (shape (nonzero (gt t 0.0))) '(5 2))
    (let ([it (tensor '((1 2) (3 4)))])
      (check-equal? (dtype (where (eq it 1) it (add1 (expt 2 53)))) 'int64)
      (check-equal? (tensor->list (where (eq it 1) it (add1 (expt 2 53))))
                    (list 1 (add1 (expt 2 53)) (add1 (expt 2 53))
                          (add1 (expt 2 53)))))
    (let ([m (ne (tensor '(1 0 1)) 0)])
      (check-equal? (dtype (where m m 2)) 'int64)
      (check-equal? (tensor->list (where m m (add1 (expt 2 53))))
                    (list 1 (add1 (expt 2 53)) 1)))
    (check-equal? (tensor->list (car (where (gt (tensor 1) 0)))) '(0))
    (check-equal? (tensor->list (car (where (gt (tensor 0) 0)))) '())
    (check-exn exn:fail:contract? (lambda () (take t 2)))
    (check-exn exn:fail:contract? (lambda () (gather t 1 '(0 1)))))

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
    (check-equal? (tensor->list (eq a (tensor '(1 5 3)))) '(1.0 0.0 1.0))
    (check-equal? (tensor->list (lt a (tensor '(2 2 2)))) '(1.0 0.0 0.0))
    (check-equal? (tensor->list (ge a 2)) '(0.0 1.0 1.0))
    (check-equal? (tensor->list (ne a 2)) '(1.0 0.0 1.0))
    (check-equal? (tensor->list (gt a 2.5)) '(0.0 0.0 1.0)))

  (test-case "flatten collapses dims and rejects an invalid range"
    (define t (reshape (arange 120) 2 3 4 5))
    (check-equal? (tensor-shape (flatten t)) '(120))
    (check-equal? (tensor-shape (flatten t 1)) '(2 60))
    (check-equal? (tensor-shape (flatten t 1 2)) '(2 12 5))
    (check-equal? (tensor-shape (flatten t 0 -1)) '(120))
    (check-equal? (tensor-shape (flatten (zeros))) '(1))
    (check-equal? (flatten '(1 (2 3))) '(1 2 3))
    (check-exn #rx"invalid dim range" (lambda () (flatten t 3 1))))

  (test-case "narrow returns a view aliasing the source storage"
    (define t (tensor '(1.0 2.0 3.0 4.0)))
    (define v (narrow t 0 1 2))
    (check-equal? (tensor->list v) '(2.0 3.0))
    (uniform! v 0.0 0.0)
    (check-equal? (tensor->list t) '(1.0 0.0 0.0 4.0))))
