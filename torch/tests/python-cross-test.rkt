#lang racket/base

;; Run: raco test torch/tests/python-cross-test.rkt (inside `nix develop`,
;; which provides Python torch; SKIPS when python3 can't import torch).

(module+ test
  (require rackunit
           "../main.rkt"
           "../nn.rkt"
           (only-in "../data/mnist.rkt" load-mnist-fixture)
           (only-in "../data/text.rkt"
                    contiguous-blocks encode load-text-fixture text->vocab)
           "private/python-env.rkt")

  (define (check-parity rel-path compute)
    (define j (python-result rel-path))
    (define py-values (hash-ref j 'values))
    (define t (compute))
    (check-equal? (tensor-shape t) (hash-ref j 'shape)
                  (format "~a: shape parity" rel-path))
    (define rkt-values (tensor->list t))
    (check-equal? (length rkt-values) (length py-values)
                  (format "~a: value count" rel-path))
    (for ([r (in-list rkt-values)]
          [p (in-list py-values)]
          [i (in-naturals)])
      (check-= r p tol (format "~a: value ~a parity" rel-path i)))
    (check-equal? (tensor->repr t) (hash-ref j 'repr)
                  (format "~a: repr parity" rel-path)))

  (cond
    [(not (python-torch-available?))
     (printf "[python-cross-test] skipped: python3 `torch` not available ~a\n"
             "(run inside `nix develop`)")]
    [else
     (check-parity "python/00_randn.py"
                   (lambda ()
                     (manual-seed! 0)
                     (randn 2 2)))
     (check-parity "python/01_arith.py"
                   (lambda ()
                     (define x (tensor '((1.0 -2.0) (3.0 -4.0))))
                     (* (+ x 1) (relu x))))
     (check-parity "python/02_matmul.py"
                   (lambda ()
                     (define a (reshape (arange 6) 2 3))
                     (@ a (t a 0 1))))
     (check-parity "python/03_autograd.py"
                   (lambda ()
                     (define x (tensor '(1.0 2.0 3.0) #:requires-grad? #t))
                     (backward! (~> x (* x) Σ))
                     (grad x)))
     (let* ([j (python-result "python/summarized_reprs.py")]
            [py (hash-ref j 'reprs)]
            [rkt (list
                  (tensor->repr (tensor (build-list 2000 values)))
                  (tensor->repr (zeros 1024 1024))
                  (tensor->repr (zeros 3 4 500))
                  (tensor->repr (zeros 1024 2 2))
                  (tensor->repr (zeros 1001))
                  (tensor->repr (eq (tensor '(1 2)) 1))
                  (tensor->repr (eq (zeros 2000) 1.0))
                  (tensor->repr
                   (tensor (build-list 2000 (lambda (i) (* i 1000000)))))
                  (tensor->repr
                   (tensor (build-list 30 (lambda (i) (* i 1000000)))))
                  (tensor->repr (zeros 6 6 6 5))
                  (tensor->repr (zeros 2 18))
                  (tensor->repr (zeros 30))
                  (tensor->repr (full 100.0 30))
                  (tensor->repr (full +inf.0 30))
                  (tensor->repr
                   (tensor (build-list
                            2000
                            (lambda (i) (* (add1 i) 100000000)))))
                  (tensor->repr (tensor '(1e10 2.5e10 -3e-7)))
                  (tensor->repr (tensor '(1e8)))
                  (tensor->repr (tensor '(+nan.0 5.0)))
                  (tensor->repr
                   (tensor (for/list ([i (in-range 2000)])
                             (* (exact->inexact (add1 i))
                                12345.6789))))
                  (tensor->repr (zeros 7 7 7 7 7)))])
       (check-equal? (length rkt) (length py)
                     "summarized-repr form count")
       (check-equal? (tensor->list (to-dtype (tensor '(16777217 1))
                                             'float64))
                     (hash-ref j 'f64_values))
       (for ([r (in-list rkt)]
             [p (in-list py)]
             [i (in-naturals)])
         (check-equal? r p (format "summarized repr ~a parity" i))))
     (check-parity "python/int64_inference.py"
                   (lambda ()
                     (define x (tensor '((1 2) (3 4))))
                     (@ x x)))
     ;; ref spec forms, in lockstep with the twin's `forms` list
     (let* ([j (python-result "python/indexing_parity.py")]
            [py (hash-ref j 'forms)]
            [ix (tensor '((1 2 3) (4 5 6)))]
            [c (arange 6)]
            [cube (tensor '(((1 2) (3 4)) ((5 6) (7 8))))]
            [form (lambda (x)
                    (list (shape x)
                          (for/list ([v (in-list (tensor->list x))])
                            (exact->inexact v))))]
            [rkt (list (form (ref ix 0))
                       (form (ref ix (: 1 3)))
                       (form (ref c (:: 1 5 2)))
                       (form (ref c (: _ _ 2)))
                       (form (ref ix : (:~ 1)))
                       (form (ref ix .. 0))
                       (form (ref ix : _))
                       (form (ref cube 1 .. 0))
                       (form (ref ix (gt ix 4)))
                       (form (ref ix (ne (tensor '(0 1)) 0)))
                       (form (ref ix '(-1 0)))
                       (form (ref cube (ne (tensor '((1 0) (0 1))) 0)))
                       (form (ref ix '(0 0 1)))
                       (form (ref ix '(1 0) 0))
                       (form (ref c (to-dtype (tensor '(4 0)) 'int64)))
                       (exact->inexact (ref ix 1 2))
                       (form (car (where (gt ix 4))))
                       (form (cadr (where (gt ix 4))))
                       (form (gather ix 1 (to-dtype (tensor '((0 2) (1 0)))
                                                    'int64)))
                       (form (take ix '(0 5 3)))
                       (form (take-along-dim
                              ix
                              (to-dtype (tensor '(5 0 2)) 'int64)))
                       (form (take ix '(-1 0 -6)))
                       (form (car (where (gt (tensor 1) 0))))
                       (form (car (where (gt (tensor 0) 0))))
                       (form (where (gt ix 2) ix (zeros 2 3)))
                       (form (where (gt ix 2) ix -1))
                       (form (where (gt ix 2) -5 ix))
                       (form (where (gt ix 2) ix -1.5))
                       (form (where (gt ix 2) -5.5 ix))
                       (form (where (gt ix 2) 1 0))
                       (form (where (gt ix 2) 1.5 0.5)))])
       (check-equal? (length rkt) (length py) "indexing form count")
       (for ([r (in-list rkt)]
             [p (in-list py)]
             [i (in-naturals)])
         (check-equal? r p (format "indexing form ~a parity" i))))
     ;; ref! / write-op forms, in lockstep with the twin's `forms` list
     (let* ([j (python-result "python/writes_parity.py")]
            [py (hash-ref j 'forms)]
            [fresh (lambda () (tensor '((1.0 2.0 3.0) (4.0 5.0 6.0))))]
            [i64 (lambda (v) (to-dtype (tensor v) 'int64))]
            [form (lambda (x)
                    (list (shape x)
                          (for/list ([v (in-list (tensor->list x))])
                            (exact->inexact v))))]
            [written (lambda (mutate!)
                       (define t (fresh))
                       (mutate! t)
                       (form t))]
            [rkt (list
                  (written (lambda (t) (ref! t 0 9)))
                  (written (lambda (t) (ref! t 1 2 0)))
                  (written (lambda (t) (ref! t -1 -1 0)))
                  (written (lambda (t) (ref! t (: 1 3) 7)))
                  (written (lambda (t) (ref! t : (: _ _ 2) 0)))
                  (written (lambda (t) (ref! t .. 0 5)))
                  (written (lambda (t) (ref! t : 1 (tensor '(8.0 9.0)))))
                  (written (lambda (t) (ref! t 0 (tensor '(7.0 8.0 9.0)))))
                  (let ([z (tensor 5.0)])
                    (ref! z .. 7)
                    (form (reshape z 1)))
                  (written (lambda (t) (ref! t (gt t 4.0) -1)))
                  (written (lambda (t) (ref! t (gt t 4.0) (tensor 0.0))))
                  (written (lambda (t)
                             (ref! t (gt t 3.0) (tensor '(7.0 8.0 9.0)))))
                  (written (lambda (t) (ref! t (ne (tensor '(1 0)) 0) 0)))
                  (written (lambda (t)
                             (ref! t (ne (tensor '(0 1)) 0)
                                   (tensor '(7.0 8.0 9.0)))))
                  (written (lambda (t)
                             (ref! t (ne (tensor '(1 0)) 0) (tensor '(7.0)))))
                  (written (lambda (t)
                             (ref! t (ne (tensor '(1 1)) 0)
                                   (tensor '((7.0) (8.0))))))
                  (written (lambda (t) (index-fill! t 0 (i64 '(1)) 3.5)))
                  (written (lambda (t)
                             (index-copy! t 0 (i64 '(0))
                                          (tensor '((9.0 9.0 9.0))))))
                  (written (lambda (t)
                             (index-add! t 0 (i64 '(0))
                                         (tensor '((1.0 1.0 1.0)))
                                         #:alpha 2)))
                  (written (lambda (t) (scatter! t 1 (i64 '((0) (2))) 5)))
                  (written (lambda (t)
                             (scatter! t 1 (i64 '((0) (2)))
                                       (tensor '((70.0) (80.0))))))
                  (written (lambda (t)
                             (scatter-add! t 1 (i64 '((0) (0)))
                                           (tensor '((1.0) (1.0))))))
                  (written (lambda (t) (masked-fill! t (gt t 5.0) 0)))
                  (written (lambda (t)
                             (masked-scatter! t (gt t 4.0)
                                              (tensor '(9.0 10.0)))))
                  (written (lambda (t)
                             (ref! t (gt t 4.0) (tensor '((9.0 10.0))))))
                  (written (lambda (t)
                             (ref! t (gt t 4.0) (tensor '(((7.0)))))))
                  (written (lambda (t)
                             (ref! t : 0 (tensor '((8.0 9.0))))))
                  (written (lambda (t)
                             (ref! t 0 (tensor '(((7.0 8.0 9.0))))))))])
       (check-equal? (length rkt) (length py) "write form count")
       (for ([r (in-list rkt)]
             [p (in-list py)]
             [i (in-naturals)])
         (check-equal? r p (format "write form ~a parity" i))))
     (let ()
       (define-module mlp (d-in d-hidden d-out)
         #:submodules ([fc1 (Linear d-in d-hidden)]
                       [fc2 (Linear d-hidden d-out)])
         #:forward (x)
         (fc2 (relu (fc1 x))))
       (define j (python-result "python/04_mlp.py"))
       (manual-seed! 0)
       (define net (mlp 4 8 2))
       (define x (randn 16 4))
       (define y (randn 16 2))
       (define opt (sgd (parameters net) #:lr 0.1))
       (define losses
         (for/list ([_ (in-range 5)])
           (zero-grads! opt)
           (define loss (mse-loss (net x) y))
           (backward! loss)
           (step! opt)
           (item loss)))
       (for ([r (in-list losses)]
             [p (in-list (hash-ref j 'losses))]
             [i (in-naturals)])
         (check-= r p tol (format "04_mlp: loss at step ~a" i)))
       (define flat-params
         (cat (for/list ([p (in-list (parameters net))])
                (reshape p -1))))
       (check-equal? (tensor-shape flat-params) (hash-ref j 'shape)
                     "04_mlp: parameter count")
       (for ([r (in-list (tensor->list flat-params))]
             [p (in-list (hash-ref j 'values))]
             [i (in-naturals)])
         (check-= r p tol (format "04_mlp: parameter ~a parity" i))))
     (let ()
       ;; Re-declared (torch/ can't reach examples/ once installed by copy in
       ;; the nix build): MUST stay in sync with examples/racket/05-mnist.rkt
       ;; — model, seed, steps, lr, full-batch regime. The shape guard below
       ;; catches structural drift; a forward/recipe edit surfaces as a
       ;; loss-parity mismatch. Same contract for the 06 gpt twin below.
       (define-module convnet ()
         #:submodules ([c1 (Conv2d 1 16 3)]
                       [c2 (Conv2d 16 32 3)]
                       [f1 (Linear 800 128)]
                       [f2 (Linear 128 10)])
         #:forward (x)
         (~> x
             c1 relu (max-pool2d 2)
             c2 relu (max-pool2d 2)
             (flatten 1) f1 relu
             f2))
       (check-equal? (map tensor-shape (parameters (convnet)))
                     '((16 1 3 3) (16) (32 16 3 3) (32)
                       (128 800) (128) (10 128) (10))
                     "convnet shape must match examples/racket/05-mnist.rkt")
       (define (train-on device)
         (with-default-device device
           (manual-seed! 0)
           (define-values (xs ys) (load-mnist-fixture))
           (define net (convnet))
           (define opt (adam (parameters net) #:lr 0.001))
           (define losses
             (for/list ([_ (in-range 5)])
               (zero-grads! opt)
               (define loss (cross-entropy (net xs) ys))
               (backward! loss)
               (step! opt)
               (item loss)))
           (values losses
                   (cat (for/list ([p (in-list (parameters net))])
                          (reshape p -1))))))
       (check-training-twin "05_mnist" "python/05_mnist.py" train-on 'cpu tol)
       (when (and (cuda-available?)
                  (python-cuda-available?))
         (check-training-twin "05_mnist" "python/05_mnist.py" train-on
                              'cuda 5e-3)))
     (let ()
       (define-module gpt-block (n-embd n-head)
         #:coerce ([n-head (if (zero? (remainder n-embd n-head))
                               n-head
                               (error 'gpt-block
                                      "n-embd ~a not divisible by n-head ~a"
                                      n-embd n-head))])
         #:submodules ([ln1 (LayerNorm n-embd)]
                       [wq (Linear n-embd n-embd)]
                       [wk (Linear n-embd n-embd)]
                       [wv (Linear n-embd n-embd)]
                       [wo (Linear n-embd n-embd)]
                       [ln2 (LayerNorm n-embd)]
                       [fc1 (Linear n-embd (* 4 n-embd))]
                       [fc2 (Linear (* 4 n-embd) n-embd)])
         #:forward (x)
         (with-default-device (tensor-device x)
           (define shape (tensor-shape x))
           (define batch (car shape))
           (define seq-len (cadr shape))
           (define head-dim (quotient n-embd n-head))
           (define (split-heads m)
             (transpose (reshape m batch seq-len n-head head-dim) 1 2))
           (define xn (ln1 x))
           (define q (split-heads (wq xn)))
           (define k (split-heads (wk xn)))
           (define v (split-heads (wv xn)))
           (define scores (div (matmul q (transpose k 2 3)) (sqrt head-dim)))
           (define causal (eq (tril (ones seq-len seq-len)) 0))
           (define att (softmax (masked-fill scores causal -inf.0) -1))
           (define ctx
             (reshape (transpose (matmul att v) 1 2) batch seq-len n-embd))
           (define x1 (add x (wo ctx)))
           (add x1 (fc2 (gelu (fc1 (ln2 x1)))))))
       (define-module gpt (vocab-size block-size)
         #:submodules ([tok-emb (Embedding vocab-size 32)]
                       [pos-emb (Embedding block-size 32)]
                       [blocks (Sequential (gpt-block 32 4) (gpt-block 32 4))]
                       [ln-f (LayerNorm 32)]
                       [head (Linear 32 vocab-size)])
         #:forward (idx)
         (with-default-device (tensor-device idx)
           (define seq-len (cadr (tensor-shape idx)))
           (define pos (to-dtype (arange seq-len) 'int64))
           (~> (add (tok-emb idx) (pos-emb pos))
               blocks ln-f head)))
       (define text (load-text-fixture))
       (define vocab (text->vocab text))
       (define v-size (vector-length vocab))
       (define block-shapes
         '((32) (32)
           (32 32) (32) (32 32) (32)
           (32 32) (32) (32 32) (32)
           (32) (32)
           (128 32) (128) (32 128) (32)))
       (check-equal? (map tensor-shape (parameters (gpt v-size 16)))
                     (append (list (list v-size 32) '(16 32))
                             block-shapes block-shapes
                             (list '(32) '(32) (list v-size 32)
                                   (list v-size)))
                     "gpt shape must match examples/racket/06-gpt.rkt")
       (define (train-on device)
         (with-default-device device
           (manual-seed! 0)
           (define-values (xs ys) (contiguous-blocks (encode vocab text) 16))
           (define net (gpt v-size 16))
           (define opt (adam (parameters net) #:lr 0.001))
           (define losses
             (for/list ([_ (in-range 5)])
               (zero-grads! opt)
               (define loss (cross-entropy (reshape (net xs) -1 v-size)
                                           (reshape ys -1)))
               (backward! loss)
               (step! opt)
               (item loss)))
           (values losses
                   (cat (for/list ([p (in-list (parameters net))])
                          (reshape p -1))))))
       (check-training-twin "06_gpt" "python/06_gpt.py" train-on 'cpu tol)
       (when (and (cuda-available?)
                  (python-cuda-available?))
         (check-training-twin "06_gpt" "python/06_gpt.py" train-on
                              'cuda 5e-3)))
     (let ()
       (define j (python-check "conv2d_init.py"))
       (manual-seed! 0)
       (define net (Conv2d 1 8 3))
       (define ps (parameters net))
       (check-equal? (map tensor-shape ps) (hash-ref j 'shapes)
                     "conv2d init: parameter shapes match nn.Conv2d")
       (define rkt-vals (apply append (map tensor->list ps)))
       (define py-vals (hash-ref j 'values))
       (check-equal? (length rkt-vals) (length py-vals)
                     "conv2d init: value count")
       (for ([r (in-list rkt-vals)] [p (in-list py-vals)] [i (in-naturals)])
         (check-= r p tol (format "conv2d init: value ~a parity" i))))
     (let ()
       (define j (python-check "embedding_init.py"))
       (manual-seed! 0)
       (define net (Embedding 7 4))
       (define w (car (parameters net)))
       (check-equal? (tensor-shape w) (hash-ref j 'shape)
                     "embedding init: weight shape matches nn.Embedding")
       (for ([r (in-list (tensor->list w))]
             [p (in-list (hash-ref j 'values))]
             [i (in-naturals)])
         (check-= r p tol (format "embedding init: value ~a parity" i))))
     (let ()
       (define j (python-check "layer_norm_forward.py"))
       (manual-seed! 0)
       (define ln (LayerNorm 5))
       (define r (ln (randn 3 5)))
       (check-equal? (tensor-shape r) (hash-ref j 'shape)
                     "layer-norm forward: shape parity")
       (for ([a (in-list (tensor->list r))]
             [b (in-list (hash-ref j 'values))]
             [i (in-naturals)])
         (check-= a b tol (format "layer-norm forward: value ~a parity" i))))
     (let ()
       (define j (python-check "pool_default_stride.py"))
       (manual-seed! 0)
       (define x (randn 1 1 4 4))
       (define mp (max-pool2d x 2))
       (define ap (avg-pool2d x 2))
       (for ([a (in-list (tensor->list mp))] [b (in-list (hash-ref j 'mp))]
             [i (in-naturals)])
         (check-= a b tol (format "max-pool2d default-stride parity ~a" i)))
       (for ([a (in-list (tensor->list ap))] [b (in-list (hash-ref j 'ap))]
             [i (in-naturals)])
         (check-= a b tol (format "avg-pool2d default-stride parity ~a" i))))
     (let ()
       (define jf (python-check "flatten.py"))
       (manual-seed! 0)
       (define x (randn 2 3 4))
       (define r (flatten x 1))
       (check-equal? (tensor-shape r) (hash-ref jf 'shape) "flatten parity: shape")
       (for ([a (in-list (tensor->list r))] [b (in-list (hash-ref jf 'values))]
             [i (in-naturals)])
         (check-= a b tol (format "flatten parity ~a" i))))
     (let ()
       (define jg (python-check "gelu.py"))
       (manual-seed! 0)
       (define xg (randn 2 3))
       (define rg (gelu xg))
       (check-equal? (tensor-shape rg) (hash-ref jg 'shape)
                     "gelu parity: shape")
       (for ([a (in-list (tensor->list rg))]
             [b (in-list (hash-ref jg 'values))]
             [i (in-naturals)])
         (check-= a b tol (format "gelu parity ~a" i))))
     (define jm (python-check "causal_mask.py"))
     (manual-seed! 0)
     (define scores (randn 2 4 4))
     (define mask (eq (tril (ones 4 4)) 0))
     (define r (softmax (masked-fill scores mask -inf.0) -1))
     (check-equal? (tensor-shape r) (hash-ref jm 'shape)
                   "causal mask parity: shape")
     (for ([a (in-list (tensor->list r))]
           [b (in-list (hash-ref jm 'values))]
           [i (in-naturals)])
       (check-= a b tol (format "causal mask parity ~a" i)))]))
