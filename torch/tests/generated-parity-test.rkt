#lang racket/base

;; The generated-op parity battery: live cross-validation of every op in the
;; codegen manifest against upstream PyTorch, plus the override drives that
;; cover optional-argument paths the default recipes leave absent. Split out
;; of python-cross-test.rkt (#29); the literate-example twins and the
;; hand-written reference checks stay there.
;;
;; The `nix develop` shell provides Python `torch`; when python3 can't
;; `import torch` (e.g. the sandboxed `nix build`, or the lean `.#ci` shell)
;; this test SKIPS -- keeping `raco test` and `nix build` green without it.
;;
;; Run for real:  raco test torch/tests/generated-parity-test.rkt

(module+ test
  (require json
           racket/port
           racket/runtime-path
           racket/string
           racket/system
           rackunit
           "../main.rkt"
           "private/python-env.rkt")

  (define-runtime-path generated-manifest "generated-parity.rktd")
  (define-runtime-path generated-rkt "../generated.rkt")

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
  ;;   (bool-tensor (v ...))       a literal bool tensor (masked_fill masks);
  ;;                               built via `ne 0` on the Racket side, so the
  ;;                               handle is genuinely bool, as ATen requires
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
          ;; comparisons (tensor + scalar rhs) -> bool masks (the read path
          ;; floatifies the values; the handles stay bool)
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
                        (optional-int64 #f))
          ;; dropout with train=#f is the identity (deterministic); the train
          ;; path is stochastic and parity-checked via the eval-mode result.
          'dropout '((tensor 2 3) (double 0.5) (bool #f))
          ;; copy_ overwrites self with src, so the seeded result equals src.
          'copy! '((tensor 2 3) (tensor 2 3) (bool #f))
          ;; --- tranche 3: the transformer op closure ---
          ;; embedding: randn weight table, literal int64 indices (with a
          ;; repeat, exercising the gather), no padding/scaling/sparse.
          'embedding '((tensor 5 3) (int-tensor (0 2 4 1)) (int64 -1)
                       (bool #f) (bool #f))
          ;; layer_norm over the last dim with affine weight+bias present
          ;; (both randn, drawn after the input so the streams align).
          'layer-norm '((tensor 2 3) (int-array (3)) (optional-tensor 3)
                        (optional-tensor 3) (double 1e-5) (bool #t))
          ;; masked_fill: a finite fill value (the -inf causal-mask idiom is
          ;; hand-checked at the facade level in python-cross-test.rkt;
          ;; number->string of -inf.0 isn't valid Python).
          'masked-fill-scalar '((tensor 6) (bool-tensor (0 1 0 1 0 1))
                                (double -100.0))
          ;; causal-mask builders at the main diagonal (offsets driven below).
          'tril '((tensor 4 4) (int64 0))
          'triu '((tensor 4 4) (int64 0))))

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
      ;; a present optional tensor of all-ones (draws no RNG, so seed
      ;; alignment holds; stable magnitude unlike a randn weight).
      [(optional-tensor-ones) (apply ones (cdr spec))]
      [(int-tensor) (to-dtype (tensor (cadr spec)) 'int64)]
      [(bool-tensor) (ne (tensor (cadr spec)) 0)]
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
      [(optional-tensor-ones) (format "torch.ones(~a)" (csv (cdr spec)))]
      [(int-tensor)
       (format "torch.tensor([~a], dtype=torch.int64)" (csv (cadr spec)))]
      [(bool-tensor)
       (format "torch.tensor([~a], dtype=torch.bool)" (csv (cadr spec)))]
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
    ;; py-name is the full aten overload path (e.g. sum.dim_IntList,
    ;; add_.Tensor) — call it explicitly, no overload guessing.
    (define callee py-name)
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
      (with-python-env
       (parameterize ([current-output-port out]
                      [current-error-port (open-output-nowhere)])
         (system* python "-c" code))))
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
                             name "generated-parity-test.rkt")))))
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
     (printf "[generated-parity-test] skipped: python3 `torch` not available ~a\n"
             "(run inside `nix develop`)")]
    [else
     ;; generated surface — every op in the codegen manifest
     (define manifest (with-input-from-file generated-manifest read))
     (for-each check-generated-parity manifest)
     ;; avg-pool2d's default recipe leaves divisor_override absent (nullopt);
     ;; drive the optional-int64 *present* path too, or its marshalling is
     ;; never compared to PyTorch.
     (check-generated-parity
      (assq 'avg-pool2d manifest)
      '((tensor 1 1 4 4) (int-array (2 2)) (int-array (2 2))
        (int-array (0 0)) (bool #f) (bool #t) (optional-int64 2))
      "[divisor=2]")
     ;; the loss recipes leave weight absent; drive the optional-tensor
     ;; weight-present branch too (ones weight: stable, exercises weight!=null).
     (check-generated-parity
      (assq 'nll-loss manifest)
      '((tensor 4 3) (int-tensor (0 2 1 0)) (optional-tensor-ones 3)
        (int64 1) (int64 -100))
      "[weight]")
     (check-generated-parity
      (assq 'cross-entropy-loss manifest)
      '((tensor 4 3) (int-tensor (0 2 1 0)) (optional-tensor-ones 3)
        (int64 1) (int64 -100) (double 0.0))
      "[weight]")
     ;; reduction=0 (None) returns per-sample losses (shape (N,)) instead
     ;; of a scalar — catches a mis-wired reduction enum as a shape change.
     (check-generated-parity
      (assq 'nll-loss manifest)
      '((tensor 4 3) (int-tensor (0 2 1 0)) (optional-tensor #f)
        (int64 0) (int64 -100))
      "[none]")
     (check-generated-parity
      (assq 'cross-entropy-loss manifest)
      '((tensor 4 3) (int-tensor (0 2 1 0)) (optional-tensor #f)
        (int64 0) (int64 -100) (double 0.0))
      "[none]")
     ;; conv2d's recipe has bias present; cover the common bias=None path.
     (check-generated-parity
      (assq 'conv2d manifest)
      '((tensor 1 1 5 5) (tensor 2 1 3 3) (optional-tensor #f)
        (int-array (1 1)) (int-array (0 0)) (int-array (1 1)) (int64 1))
      "[no-bias]")
     ;; the dim-wise reductions only drive dim-present; cover the absent
     ;; (full-reduction) path against PyTorch too.
     (check-generated-parity
      (assq 'sum-dim-intlist manifest)
      '((tensor 2 3) (optional-int-array #f) (bool #f) (dtype #f))
      "[full]")
     (check-generated-parity
      (assq 'mean-dim manifest)
      '((tensor 2 3) (optional-int-array #f) (bool #f) (dtype #f))
      "[full]")
     ;; default recipes use keepdim=#f; cover keepdim=#t (kept dim) too.
     (check-generated-parity
      (assq 'sum-dim-intlist manifest)
      '((tensor 2 3) (optional-int-array (1)) (bool #t) (dtype #f))
      "[keepdim]")
     (check-generated-parity
      (assq 'mean-dim manifest)
      '((tensor 2 3) (optional-int-array (1)) (bool #t) (dtype #f))
      "[keepdim]")
     ;; layer_norm's default recipe has affine weight+bias present; cover
     ;; the bare (no-affine) path — both optionals nullopt.
     (check-generated-parity
      (assq 'layer-norm manifest)
      '((tensor 2 3) (int-array (3)) (optional-tensor #f)
        (optional-tensor #f) (double 1e-5) (bool #t))
      "[no-affine]")
     ;; embedding with a real padding_idx: forward values are insensitive
     ;; to it in ATen, so this pins the non-default marshalling path (the
     ;; recipe's -1 is the #f mapping, never a real index).
     (check-generated-parity
      (assq 'embedding manifest)
      '((tensor 5 3) (int-tensor (0 2 4 1)) (int64 2) (bool #f) (bool #f))
      "[padding-idx=2]")
     ;; layer_norm over trailing [2,3] dims jointly — the multi-dim
     ;; normalized-shape the facade contract advertises.
     (check-generated-parity
      (assq 'layer-norm manifest)
      '((tensor 2 2 3) (int-array (2 3)) (optional-tensor 2 3)
        (optional-tensor 2 3) (double 1e-5) (bool #t))
      "[multi-dim]")
     ;; tril/triu at offset diagonals (the GPT causal mask uses tril at 0;
     ;; the offsets pin the diagonal argument's sign convention).
     (check-generated-parity
      (assq 'tril manifest)
      '((tensor 4 4) (int64 -1))
      "[diag=-1]")
     (check-generated-parity
      (assq 'triu manifest)
      '((tensor 4 4) (int64 1))
      "[diag=1]")]))
