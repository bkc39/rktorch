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
;; Run for real:  raco test torch/tests/python-cross-test.rkt

(module+ test
  (require racket/port
           racket/runtime-path
           racket/string
           racket/system
           json
           rackunit
           "../main.rkt"
           ;; conv2d/max-pool2d/flatten name both the functional ops (under
           ;; torch, used here) and the nn layers; this suite exercises the
           ;; functional surface, so drop the colliding layer names from nn.
           (except-in "../nn.rkt" conv2d max-pool2d flatten))

  (define-runtime-path examples-dir "../../examples")
  (define-runtime-path generated-manifest "generated-parity.rktd")
  (define-runtime-path generated-rkt "../generated.rkt")

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

  ;; --- generated-op parity battery ---------------------------------------
  ;; The codegen manifest (written by `python3 -m codegen`) names every
  ;; generated op; each op needs an input recipe here. An op without a
  ;; recipe fails loudly, so extending codegen/allowlist.txt forces a
  ;; conscious choice of parity inputs. The reference call goes through
  ;; `torch.ops.aten.<attr>` (the faithful at:: schema -- functional ops
  ;; like adaptive_avg_pool2d that aren't on the torch top-level still
  ;; resolve, and arg order matches the C++ shim). In-place ops call
  ;; `torch.ops.aten.<attr>_` and read back the mutated receiver.
  ;;
  ;; Recipe specs:
  ;;   (tensor dim ...)            seeded randn
  ;;   (tensors (dim ...) ...)     a list of seeded randns
  ;;   (optional-tensor dim ...)   seeded randn; (optional-tensor #f) -> None
  ;;   (int64 v) (double v) (bool v) (int-array (v ...))   literals
  ;;   (int-tensor (v ...))        a literal int64 tensor (loss targets)
  ;;   (optional-int64 v|#f) (optional-int-array (v ...)|#f)  optional, #f=None
  ;;   (dtype sym|#f)              a ScalarType (kwarg-only on the aten side)
  ;;   (kwarg "name" v)            a scalar passed positionally to the Racket
  ;;                               op but as name=v to the kwarg-only aten arg
  (define generated-recipes
    (hash 'matmul '((tensor 2 3) (tensor 3 2))
          'mm '((tensor 2 2) (tensor 2 2))
          'mv '((tensor 2 3) (tensor 3))
          'dot '((tensor 4) (tensor 4))
          'reshape '((tensor 2 3) (int-array (3 2)))
          'cat '((tensors (2 3) (2 3)) (int64 0))
          'narrow '((tensor 4) (int64 0) (int64 1) (int64 2))
          ;; conv + pooling
          'conv2d '((tensor 1 1 5 5) (tensor 2 1 3 3) (optional-tensor 2)
                    (int-array (1 1)) (int-array (0 0)) (int-array (1 1))
                    (int64 1))
          'max-pool2d '((tensor 1 1 4 4) (int-array (2 2)) (int-array (2 2))
                        (int-array (0 0)) (int-array (1 1)) (bool #f))
          'adaptive-avg-pool2d '((tensor 1 1 4 4) (int-array (2 2)))
          ;; in-place family (receiver mutated, then read)
          'add-tensor! '((tensor 3) (tensor 3) (kwarg "alpha" 2.0))
          'mul-tensor! '((tensor 3) (tensor 3))
          'addcmul! '((tensor 3) (tensor 3) (tensor 3) (kwarg "value" 0.5))
          'addcdiv! '((tensor 3) (tensor 3) (tensor 3) (kwarg "value" 0.5))
          'lerp-tensor! '((tensor 3) (tensor 3) (tensor 3))
          ;; comparisons (tensor + scalar rhs) -> float32 masks
          'eq-tensor '((tensor 3) (tensor 3))
          'eq-scalar '((tensor 3) (double 0.0))
          'ne-tensor '((tensor 3) (tensor 3))
          'ne-scalar '((tensor 3) (double 0.0))
          'lt-tensor '((tensor 3) (tensor 3))
          'lt-scalar '((tensor 3) (double 0.0))
          'le-tensor '((tensor 3) (tensor 3))
          'le-scalar '((tensor 3) (double 0.0))
          'gt-tensor '((tensor 3) (tensor 3))
          'gt-scalar '((tensor 3) (double 0.0))
          'ge-tensor '((tensor 3) (tensor 3))
          'ge-scalar '((tensor 3) (double 0.0))
          ;; losses: randn logits, literal int64 targets, no weight, Mean(=1)
          'nll-loss '((tensor 4 3) (int-tensor (0 2 1 0)) (optional-tensor #f)
                      (int64 1) (int64 -100))
          'cross-entropy-loss '((tensor 4 3) (int-tensor (0 2 1 0))
                                (optional-tensor #f) (int64 1) (int64 -100)
                                (double 0.0))
          ;; dim-wise reductions: dim=[1], keepdim=#f, dtype=None
          'sum-dim-intlist '((tensor 2 3) (optional-int-array (1)) (bool #f)
                             (dtype #f))
          'mean-dim '((tensor 2 3) (optional-int-array (1)) (bool #f)
                      (dtype #f))
          ;; avg_pool2d: 2x2 window, default count_include_pad, no divisor
          'avg-pool2d '((tensor 1 1 4 4) (int-array (2 2)) (int-array (2 2))
                        (int-array (0 0)) (bool #f) (bool #t)
                        (optional-int64 #f))))

  ;; Both sides draw tensor inputs left to right from the same seed, so the
  ;; RNG streams line up exactly like the literate-example twins. Specs that
  ;; pass literals (incl. an absent optional tensor) draw nothing.
  (define (spec->racket-arg spec)
    (case (car spec)
      [(tensor) (apply randn (cdr spec))]
      [(tensors)
       (for/list ([dims (in-list (cdr spec))])
         (apply randn dims))]
      [(optional-tensor)
       (if (equal? (cdr spec) '(#f)) #f (apply randn (cdr spec)))]
      [(int-tensor) (to-dtype (tensor (cadr spec)) 'int64)]
      [(int64 double bool int-array optional-int64 optional-int-array dtype)
       (cadr spec)]
      [(kwarg) (caddr spec)]
      [else (error 'generated-parity "unknown recipe spec: ~a" spec)]))

  (define (spec->python-expr spec)
    (define (csv vs)
      (string-join (map number->string vs) ", "))
    (case (car spec)
      [(tensor) (format "torch.randn(~a)" (csv (cdr spec)))]
      [(tensors)
       (format "[~a]"
               (string-join (for/list ([dims (in-list (cdr spec))])
                              (format "torch.randn(~a)" (csv dims)))
                            ", "))]
      [(optional-tensor)
       (if (equal? (cdr spec) '(#f)) "None" (format "torch.randn(~a)"
                                                    (csv (cdr spec))))]
      [(int-tensor)
       (format "torch.tensor([~a], dtype=torch.int64)" (csv (cadr spec)))]
      [(int64 double) (number->string (cadr spec))]
      [(kwarg) (number->string (caddr spec))]
      [(bool) (if (cadr spec) "True" "False")]
      [(int-array) (format "[~a]" (csv (cadr spec)))]
      [(optional-int64) (if (cadr spec) (number->string (cadr spec)) "None")]
      [(optional-int-array)
       (if (cadr spec) (format "[~a]" (csv (cadr spec))) "None")]
      [(dtype) (if (cadr spec) (format "torch.~a" (cadr spec)) "None")]
      [else (error 'generated-parity "unknown recipe spec: ~a" spec)]))

  ;; The call argument for spec i. A kwarg-only scalar renders as name=ai;
  ;; dtype is kwarg-only on the aten reductions (sum.dim_IntList/mean.dim);
  ;; everything else is positional ai.
  (define (spec->python-call-arg spec i)
    (case (car spec)
      [(kwarg) (format "~a=a~a" (cadr spec) i)]
      [(dtype) (format "dtype=a~a" i)]
      [else (format "a~a" i)]))

  (define (generated-python-result py-name specs inplace?)
    (define callee (if inplace? (string-append py-name "_") py-name))
    (define call-args
      (string-join (for/list ([s (in-list specs)] [i (in-naturals)])
                     (spec->python-call-arg s i))
                   ", "))
    (define invoke (format "torch.ops.aten.~a(~a)" callee call-args))
    (define code
      (string-append
       "import json, torch\n"
       "torch.manual_seed(0)\n"
       (apply string-append
              (for/list ([s (in-list specs)]
                         [i (in-naturals)])
                (format "a~a = ~a\n" i (spec->python-expr s))))
       ;; in-place mutates a0 (the receiver); functional returns the result.
       (if inplace?
           (format "~a\nr = a0\n" invoke)
           (format "r = ~a\n" invoke))
       "print(json.dumps({\"shape\": list(r.shape),"
       " \"values\": [float(v) for v in r.flatten().tolist()]}))"))
    (define out (open-output-string))
    (define ok?
      (parameterize ([current-output-port out]
                     [current-error-port (open-output-nowhere)])
        (system* python "-c" code)))
    (unless ok?
      (error 'generated-parity "python failed for ~a" py-name))
    (read-json (open-input-string (get-output-string out))))

  ;; specs-override drives an extra input set through an op already in the
  ;; manifest (e.g. an optional param's present path that its default recipe
  ;; leaves absent); label disambiguates the check names.
  (define (check-generated-parity entry [specs-override #f] [label ""])
    (define name (car entry))
    (define py-name (cadr entry))
    (define kinds (caddr entry))
    ;; 4th element (inplace?) is present for manifests written by the
    ;; tranche-2 generator; default #f keeps older manifests readable.
    (define inplace? (and (>= (length entry) 4) (list-ref entry 3)))
    (define specs
      (or specs-override
          (hash-ref generated-recipes name
                    (lambda ()
                      (error 'generated-parity
                             "no input recipe for generated op ~a; add one to ~a"
                             name "python-cross-test.rkt")))))
    (check-equal? (length specs) (length kinds)
                  (format "~a~a: recipe arity matches manifest" name label))
    (define j (generated-python-result py-name specs inplace?))
    (manual-seed! 0)
    (define args (map spec->racket-arg specs))
    (define op (dynamic-require generated-rkt name))
    ;; An in-place op returns its (now-mutated) receiver, so reading the
    ;; result reads the mutation -- same as the functional path.
    (define result (apply op args))
    (check-equal? (tensor-shape result) (hash-ref j 'shape)
                  (format "~a~a: generated shape parity" name label))
    (for ([r (in-list (tensor->list result))]
          [p (in-list (hash-ref j 'values))]
          [i (in-naturals)])
      (check-= r p tol
               (format "~a~a: generated value ~a parity" name label i))))

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
         (check-= r p tol (format "04_mlp: parameter ~a parity" i))))
     ;; generated surface — every op in the codegen manifest
     (let ([manifest (with-input-from-file generated-manifest read)])
       (for-each check-generated-parity manifest)
       ;; avg-pool2d's default recipe leaves divisor_override absent (nullopt);
       ;; drive the optional-int64 *present* path too, or its marshalling is
       ;; never compared to PyTorch.
       (check-generated-parity
        (assq 'avg-pool2d manifest)
        '((tensor 1 1 4 4) (int-array (2 2)) (int-array (2 2))
          (int-array (0 0)) (bool #f) (bool #t) (optional-int64 2))
        "[divisor=2]"))]))
