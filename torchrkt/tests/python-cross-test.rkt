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
;; Tolerance, not bit-exactness: v0 builds the C++ side against libtorch-bin,
;; which may differ in patch version from the Python torch here.  Seeded CPU
;; randn is stable across recent versions, but the tolerance absorbs any drift.
;; Flip the flake's `torchSource` to "python" for guaranteed bit-exact parity.
;;
;; Run for real:  raco test torchrkt/tests/python-cross-test.rkt

(module+ test
  (require racket/port
           racket/runtime-path
           racket/system
           json
           rackunit
           "../main.rkt"
           "../nn.rkt")

  (define-runtime-path examples-dir "../../examples")

  (define python (find-executable-path "python3"))

  (define (python-torch-available?)
    (and python
         (parameterize ([current-output-port (open-output-nowhere)]
                        [current-error-port (open-output-nowhere)])
           (system* python "-c" "import torch"))))

  ;; Run a Python reference file and return its parsed JSON hash.
  (define (python-result rel-path)
    (define py (build-path examples-dir rel-path))
    (define out (open-output-string))
    (define ok?
      (parameterize ([current-output-port out]
                     [current-error-port (open-output-nowhere)])
        (system* python (path->string py))))
    (unless ok?
      (error 'python-cross-test "python failed for ~a" rel-path))
    (read-json (open-input-string (get-output-string out))))

  (define tol 1e-4)

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
                     (mul (add x 1) (relu x))))
     ;; 02 — arange/reshape/transpose/matmul
     (check-parity "python/02_matmul.py"
                   (lambda ()
                     (define a (reshape (arange 6) 2 3))
                     (matmul a (transpose a 0 1))))
     ;; 03 — autograd: d(sum(x*x))/dx == 2x
     (check-parity "python/03_autograd.py"
                   (lambda ()
                     (define x (requires-grad! (tensor '(1 2 3))))
                     (backward! (sum (mul x x)))
                     (grad x)))
     ;; 04 — the v1 capstone: seeded MLP init + 5 SGD steps track PyTorch
     ;; (losses per step and every post-training parameter). No repr check:
     ;; the 58-value parameter vector would hit PyTorch's line wrapping,
     ;; which the formatter doesn't reproduce.
     (let ()
       (define-module mlp (d-in d-hidden d-out)
         #:submodules ([fc1 (linear d-in d-hidden)]
                       [fc2 (linear d-hidden d-out)])
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
         (check-= r p tol (format "04_mlp: parameter ~a parity" i))))]))
