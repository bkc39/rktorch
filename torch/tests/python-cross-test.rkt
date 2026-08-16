#lang racket/base

;; Live cross-validation of the Racket bindings against upstream PyTorch.
;;
;; For each reference example we run the adjacent Python file (examples/python/
;; NN-*.py, which prints {"shape": [...], "values": [...]}) and check our Racket
;; result agrees within a float tolerance.  The `nix develop` shell provides
;; Python `torch`; when python3 can't `import torch` (e.g. the sandboxed
;; `nix build`, or the lean `.#ci` shell) this test SKIPS -- keeping
;; `raco test` and `nix build` green without it.
;;
;; This file holds the literate-example twins (00-05) and the hand-written
;; reference checks (torch/tests/python/*.py); the generated-op battery over
;; the codegen manifest lives in generated-parity-test.rkt (#29). Shared
;; python-subprocess infra (env pinning, runners, tolerance rationale) is in
;; private/python-env.rkt.
;;
;; Run for real:  raco test torch/tests/python-cross-test.rkt

(module+ test
  (require rackunit
           ;; layers are PascalCase (Conv2d/Linear) and the functional ops stay
           ;; lowercase on `torch` (max-pool2d/avg-pool2d/flatten), so requiring
           ;; both is collision-free (#11). This suite exercises both: the nn
           ;; Conv2d layer (seeded-init parity) and the functional pool/flatten.
           "../main.rkt"
           "../nn.rkt"
           ;; the committed 256-image fixture for the Conv-MNIST parity twin.
           (only-in "../data/mnist.rkt" load-mnist-fixture)
           ;; the committed prose fixture + char helpers for the GPT twin.
           (only-in "../data/text.rkt"
                    contiguous-blocks encode load-text-fixture text->vocab)
           "private/python-env.rkt")

  ;; Run the Python reference at rel-path and check the tensor produced by
  ;; `compute` agrees: shape exactly, values within tol, repr byte-for-byte
  ;; (the headline: racket repl == python repl).
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
     ;; 00 — seeded randn 2x2
     (check-parity "python/00_randn.py"
                   (lambda ()
                     (manual-seed! 0)
                     (randn 2 2)))
     ;; 01 — deterministic elementwise arithmetic
     (check-parity "python/01_arith.py"
                   (lambda ()
                     (define x (tensor '((1 -2) (3 -4))))
                     (* (+ x 1) (relu x))))
     ;; 02 — arange/reshape/transpose/matmul
     (check-parity "python/02_matmul.py"
                   (lambda ()
                     (define a (reshape (arange 6) 2 3))
                     (@ a (t a 0 1))))
     ;; 03 — autograd: d(sum(x*x))/dx == 2x
     (check-parity "python/03_autograd.py"
                   (lambda ()
                     (define x (tensor '(1 2 3) #:requires-grad? #t))
                     (backward! (~> x (* x) Σ))
                     (grad x)))
     ;; 04 — the v1 capstone: seeded MLP init + 5 SGD steps track PyTorch
     ;; (losses per step and every post-training parameter). No repr check:
     ;; the 58-value parameter vector would hit PyTorch's line wrapping,
     ;; which the formatter doesn't reproduce.
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
     ;; Shared capstone-twin checker (generalized from the #19 check-convnet
     ;; for the 06-gpt pass): run the Python twin at rel-path pinned to
     ;; `device` — RKTORCH_PARITY_DEVICE set via the env-copy wrapper, never
     ;; the process-wide env — and check the Racket `train-on` run on the same
     ;; device agrees: per-step losses, parameter count, every flattened
     ;; post-training parameter. train-on: device -> (values losses
     ;; flat-params). CUDA values come back to host for the comparison.
     ;; `dev-tol`: CPU is bit-stable (the strict, CI-gating `tol`); the CUDA
     ;; pass is looser because the two stacks (libtorch 2.9 vs Python torch
     ;; 2.12) pick different cuDNN/cuBLAS algorithms and reduction orders, so
     ;; values after 5 Adam steps drift ~1e-3 even though seeded init matches.
     (define (check-training-twin label rel-path train-on device dev-tol)
       (define j
         (call-with-python-env
          #:env (list (cons "RKTORCH_PARITY_DEVICE" (symbol->string device)))
          (lambda () (python-result rel-path))))
       (define-values (losses flat-params) (train-on device))
       (for ([r (in-list losses)]
             [p (in-list (hash-ref j 'losses))]
             [i (in-naturals)])
         (check-= r p dev-tol (format "~a[~a]: loss at step ~a" label device i)))
       (check-equal? (tensor-shape flat-params) (hash-ref j 'shape)
                     (format "~a[~a]: parameter count" label device))
       ;; flat-params lives on `device`; copy to host explicitly before reading
       ;; values, rather than relying on tensor->list to do it implicitly.
       (define host-params (to-device flat-params 'cpu))
       (for ([r (in-list (tensor->list host-params))]
             [p (in-list (hash-ref j 'values))]
             [i (in-naturals)])
         (check-= r p dev-tol
                  (format "~a[~a]: parameter ~a parity" label device i))))
     ;; 05 — the v2 capstone: the Conv-MNIST convnet trained on the committed
     ;; 256-image fixture. Seeded conv2d/linear init + 5 full-batch Adam steps
     ;; track PyTorch (per-step losses and every post-training parameter). The
     ;; helper trains on a chosen device; the Python twin reads its device from
     ;; RKTORCH_PARITY_DEVICE. CPU runs always (deterministic, CI-gating); the
     ;; CUDA pass runs only when a GPU is present on both sides (the "accelerator
     ;; programs match PyTorch" check) and self-skips otherwise.
     (let ()
       ;; The Conv-MNIST model for the parity pass. Re-declared here (rather than
       ;; imported from examples/racket/05-mnist.rkt) because torch/ can't reach
       ;; examples/ once installed by copy in the nix build — and kept inside the
       ;; python-available branch so its tensors aren't allocated when the suite
       ;; skips. Must stay in sync with that example's convnet.
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
       ;; Structural guard against silent divergence from the example's convnet:
       ;; a changed layer size there trips this immediately. Shape only — it does
       ;; NOT guard the forward body (activation choice, pooling order,
       ;; threading); such a change surfaces instead as a step-1 parity-value
       ;; mismatch, so treat a forward edit in the example as a cue to re-audit.
       (check-equal? (map tensor-shape (parameters (convnet)))
                     '((16 1 3 3) (16) (32 16 3 3) (32)
                       (128 800) (128) (10 128) (10))
                     "convnet shape must match examples/racket/05-mnist.rkt")
       ;; Train on `device` and return (values losses flat-params), exactly as
       ;; examples/racket/05-mnist.rkt's run-example does. Must stay in sync with
       ;; run-example: seed=0, steps=5, lr=0.001, full-batch on the fixture, no
       ;; shuffling — a mismatch here shows up as a step-0 loss-parity failure.
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
       ;; the accelerator-parity pass: only when this host has a CUDA device AND
       ;; the Python torch here was built with CUDA (the cu130 torch-bin wheel,
       ;; staged by `nix develop .#cuda`). On a CPU-only box / CPU torch it skips.
       (when (and (cuda-available?)
                  (python-cuda-available?))
         (check-training-twin "05_mnist" "python/05_mnist.py" train-on
                              'cuda 5e-3)))
     ;; 06 — the v3 capstone: the char-GPT trained on the committed 841-char
     ;; Heart of Darkness fixture, same shape as the 05 pass: seeded
     ;; embedding/linear init + 5 full-batch Adam steps track PyTorch, CPU
     ;; strict and CUDA looser/self-skipping via check-training-twin.
     (let ()
       ;; The GPT for the parity pass. Re-declared here (rather than imported
       ;; from examples/racket/06-gpt.rkt) because torch/ can't reach
       ;; examples/ once installed by copy in the nix build — and kept inside
       ;; the python-available branch so its tensors aren't allocated when the
       ;; suite skips. Must stay in sync with that example's gpt-block/gpt
       ;; (fixture scale: block-size 16, n-embd 32, n-head 4, n-layer 2).
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
       ;; Structural guard against silent divergence from the example's gpt
       ;; (the convnet-guard pattern): shape only — a forward edit in the
       ;; example surfaces as a step-0 loss-parity mismatch instead.
       (define block-shapes
         '((32) (32)                              ; ln1
           (32 32) (32) (32 32) (32)              ; wq wk
           (32 32) (32) (32 32) (32)              ; wv wo
           (32) (32)                              ; ln2
           (128 32) (128) (32 128) (32)))         ; fc1 fc2
       (check-equal? (map tensor-shape (parameters (gpt v-size 16)))
                     (append (list (list v-size 32) '(16 32))
                             block-shapes block-shapes
                             (list '(32) '(32) (list v-size 32)
                                   (list v-size)))
                     "gpt shape must match examples/racket/06-gpt.rkt")
       ;; Train on `device`, exactly as run-example does: seed=0, steps=5,
       ;; lr=0.001, full-batch 16-char blocks of the fixture. Must stay in
       ;; sync with run-example — a mismatch shows as a step-0 loss failure.
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
     ;; conv2d layer: the seeded init (kaiming-uniform weight + uniform bias,
     ;; in that order) must match nn.Conv2d.reset_parameters value-for-value,
     ;; which depends on fan-in = in*kH*kW being computed exactly like
     ;; torch.nn.init._calculate_fan_in_and_fan_out.
     (let ()
       (define j (python-check "conv2d_init.py"))
       (manual-seed! 0)
       (define net (Conv2d 1 8 3))
       (define ps (parameters net))  ; weight then bias, declaration order
       (check-equal? (map tensor-shape ps) (hash-ref j 'shapes)
                     "conv2d init: parameter shapes match nn.Conv2d")
       (define rkt-vals (apply append (map tensor->list ps)))
       (define py-vals (hash-ref j 'values))
       (check-equal? (length rkt-vals) (length py-vals)
                     "conv2d init: value count")
       (for ([r (in-list rkt-vals)] [p (in-list py-vals)] [i (in-naturals)])
         (check-= r p tol (format "conv2d init: value ~a parity" i))))
     ;; Embedding layer: the seeded standard-normal init (normal-init =
     ;; randn) must match nn.Embedding.reset_parameters (init.normal_)
     ;; value-for-value — randn is empty().normal_(), same RNG consumption.
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
     ;; LayerNorm layer: init is deterministic (ones/zeros), so the forward
     ;; on a seeded input is the meaningful parity check.
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
     ;; the promoted max/avg-pool2d wrappers default #:stride to kernel-size
     ;; (PyTorch's stride=None); the generated battery hits the raw bindings,
     ;; so parity-check that facade default against F.* directly.
     (let ()
       (define j (python-check "pool_default_stride.py"))
       (manual-seed! 0)
       (define x (randn 1 1 4 4))
       (define mp (max-pool2d x 2))  ; promoted: #:stride #f -> kernel-size
       (define ap (avg-pool2d x 2))
       (for ([a (in-list (tensor->list mp))] [b (in-list (hash-ref j 'mp))]
             [i (in-naturals)])
         (check-= a b tol (format "max-pool2d default-stride parity ~a" i)))
       (for ([a (in-list (tensor->list ap))] [b (in-list (hash-ref j 'ap))]
             [i (in-naturals)])
         (check-= a b tol (format "avg-pool2d default-stride parity ~a" i))))
     ;; flatten is Racket-side reshape logic, not a generated binding, so it's
     ;; outside the manifest battery; parity-check it against torch.flatten.
     (let ()
       (define jf (python-check "flatten.py"))
       (manual-seed! 0)
       (define x (randn 2 3 4))
       (define r (flatten x 1))
       (check-equal? (tensor-shape r) (hash-ref jf 'shape) "flatten parity: shape")
       (for ([a (in-list (tensor->list r))] [b (in-list (hash-ref jf 'values))]
             [i (in-naturals)])
         (check-= a b tol (format "flatten parity ~a" i))))
     ;; gelu is hand-written (kwarg-only `approximate` puts it outside the
     ;; codegen IR/manifest); parity-check the erf-form default against
     ;; F.gelu directly.
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
     ;; the causal-attention mask idiom, end to end: build the mask from
     ;; tril + eq (a bool tensor), fill the upper triangle of *batched*
     ;; scores with -inf (the [T,T]-mask-over-[B,T,T] broadcast the
     ;; training loop uses), and softmax — exactly what the 06-gpt
     ;; capstone's attention will do. The recipe battery can't express
     ;; -inf (not valid Python via number->string), so this facade-level
     ;; composition is hand-checked. (Bare defines, not a let block: this
     ;; is the clause's last check, so nothing below can capture them —
     ;; the earlier blocks keep their lets to scope their j/net/x names.)
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
