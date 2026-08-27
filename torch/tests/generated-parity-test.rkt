#lang racket/base

;; Run: raco test torch/tests/generated-parity-test.rkt (inside `nix develop`,
;; which provides Python torch; SKIPS when python3 can't import torch).

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

  ;; Every op in the codegen manifest needs an input recipe here (a missing
  ;; one fails loudly). Spec -> input: (tensor dim ...) = seeded randn,
  ;; (tensors ...) a list of them, (bool-tensor ...) a genuine bool handle
  ;; via `ne 0`, (kwarg "name" v) a scalar passed as name=v to a kwarg-only
  ;; aten arg, other heads are literals with #f = None.
  (define generated-recipes
    (hash 'matmul '((tensor 2 3) (tensor 3 2))
          'mm '((tensor 2 2) (tensor 2 2))
          'mv '((tensor 2 3) (tensor 3))
          'dot '((tensor 4) (tensor 4))
          'abs '((tensor 2 3))
          'broadcast-to '((tensor 1 3) (int-array (2 3)))
          'cos '((tensor 2 3))
          'sin '((tensor 2 3))
          'reshape '((tensor 2 3) (int-array (3 2)))
          'select-int '((tensor 2 3) (int64 0) (int64 1))
          'slice-tensor '((tensor 6) (int64 0) (optional-int64 1)
                          (optional-int64 5) (int64 2))
          'index-select '((tensor 4 3) (int64 0) (int-tensor (0 2)))
          'masked-select '((tensor 6) (bool-tensor (0 1 0 1 1 0)))
          'take '((tensor 2 3) (int-tensor (0 5 3)))
          'gather '((tensor 2 3) (int64 1) (int-tensor-2d ((0 2) (1 0)))
                    (kwarg "sparse_grad" #f))
          'take-along-dim '((tensor 2 3) (int-tensor-2d ((0) (2)))
                            (optional-int64 1))
          'where-self '((bool-tensor (0 1 1 0 1 0)) (tensor 6) (tensor 6))
          'where-scalarother '((bool-tensor (0 1 1 0 1 0)) (tensor 6)
                               (double -1.0))
          'where-scalarself '((bool-tensor (0 1 1 0 1 0)) (double -5.0)
                              (tensor 6))
          'where-scalar '((bool-tensor (0 1 1 0 1 0)) (double 1.0)
                          (double 0.0))
          'nonzero '((bool-tensor (0 1 1 0)))
          'cat '((tensors (2 3) (2 3)) (int64 0))
          'narrow '((tensor 4) (int64 0) (int64 1) (int64 2))
          'conv2d '((tensor 1 1 5 5) (tensor 2 1 3 3) (optional-tensor 2)
                    (int-array (1 1)) (int-array (0 0)) (int-array (1 1))
                    (int64 1))
          'max-pool2d '((tensor 1 1 4 4) (int-array (2 2)) (int-array (2 2))
                        (int-array (0 0)) (int-array (1 1)) (bool #f))
          'adaptive-avg-pool2d '((tensor 1 1 4 4) (int-array (2 2)))
          'add-tensor! '((tensor 3) (tensor 3) (kwarg "alpha" 2.0))
          'mul-tensor! '((tensor 3) (tensor 3))
          'addcmul! '((tensor 3) (tensor 3) (tensor 3) (kwarg "value" 0.5))
          'addcdiv! '((tensor 3) (tensor 3) (tensor 3) (kwarg "value" 0.5))
          'lerp-tensor! '((tensor 3) (tensor 3) (tensor 3))
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
          'nll-loss '((tensor 4 3) (int-tensor (0 2 1 0)) (optional-tensor #f)
                      (int64 1) (int64 -100))
          'cross-entropy-loss '((tensor 4 3) (int-tensor (0 2 1 0))
                                (optional-tensor #f) (int64 1) (int64 -100)
                                (double 0.0))
          'sum-dim-intlist '((tensor 2 3) (optional-int-array (1)) (bool #f)
                             (dtype #f))
          'mean-dim '((tensor 2 3) (optional-int-array (1)) (bool #f)
                      (dtype #f))
          'avg-pool2d '((tensor 1 1 4 4) (int-array (2 2)) (int-array (2 2))
                        (int-array (0 0)) (bool #f) (bool #t)
                        (optional-int64 #f))
          'dropout '((tensor 2 3) (double 0.5) (bool #f))
          'copy! '((tensor 2 3) (tensor 2 3) (bool #f))
          'fill-scalar! '((tensor 2 3) (double -3.0))
          'index-copy! '((tensor 3 3) (int64 0) (int-tensor (0 2))
                         (tensor 2 3))
          'index-fill-int-tensor! '((tensor 3 3) (int64 0)
                                    (int-tensor (0 2)) (scalar-tensor 4.5))
          'masked-fill-tensor! '((tensor 6) (bool-tensor (0 1 0 1 0 1))
                                 (scalar-tensor 4.5))
          'index-add! '((tensor 3 3) (int64 0) (int-tensor (0 2))
                        (tensor 2 3) (kwarg "alpha" 2.0))
          'index-fill-int-scalar! '((tensor 3 3) (int64 0)
                                    (int-tensor (0 2)) (double -7.0))
          'scatter-src! '((tensor 2 3) (int64 1)
                          (int-tensor-2d ((0 2) (1 0))) (tensor 2 2))
          'scatter-value! '((tensor 2 3) (int64 1)
                            (int-tensor-2d ((0 2) (1 0))) (double -9.0))
          'scatter-add! '((tensor 2 3) (int64 1)
                          (int-tensor-2d ((0 2) (1 0))) (tensor 2 2))
          'masked-fill-scalar! '((tensor 6) (bool-tensor (0 1 0 1 0 1))
                                 (double -100.0))
          'masked-scatter! '((tensor 6) (bool-tensor (0 1 0 1 0 1))
                             (tensor 6))
          'embedding '((tensor 5 3) (int-tensor (0 2 4 1)) (int64 -1)
                       (bool #f) (bool #f))
          'layer-norm '((tensor 2 3) (int-array (3)) (optional-tensor 3)
                        (optional-tensor 3) (double 1e-5) (bool #t))
          'masked-fill-scalar '((tensor 6) (bool-tensor (0 1 0 1 0 1))
                                (double -100.0))
          'tril '((tensor 4 4) (int64 0))
          'triu '((tensor 4 4) (int64 0))))

  ;; Tensor specs draw seeded randns left to right — both sides consume the
  ;; same RNG stream, so spec order and draw counts must match exactly.
  (define (spec->racket-arg spec)
    (case (car spec)
      [(tensor) (apply randn (cdr spec))]
      [(tensors)
       (for/list ([dims (in-list (cdr spec))])
         (apply randn dims))]
      [(optional-tensor)
       (if (equal? (cdr spec) '(#f)) #f (apply randn (cdr spec)))]
      [(optional-tensor-ones) (apply ones (cdr spec))]
      [(int-tensor int-tensor-2d) (to-dtype (tensor (cadr spec)) 'int64)]
      [(scalar-tensor) (tensor (cadr spec))]
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
      [(int-tensor-2d)
       (format "torch.tensor([~a], dtype=torch.int64)"
               (string-join (for/list ([row (in-list (cadr spec))])
                              (format "[~a]" (csv row)))
                            ", "))]
      [(scalar-tensor) (format "torch.tensor(~a)" (cadr spec))]
      [(bool-tensor)
       (format "torch.tensor([~a], dtype=torch.bool)" (csv (cadr spec)))]
      [(int64 double) (number->string (cadr spec))]
      [(kwarg)
       (define v (caddr spec))
       (cond
         [(boolean? v) (if v "True" "False")]
         [else (number->string v)])]
      [(bool) (if (cadr spec) "True" "False")]
      [(int-array) (format "[~a]" (csv (cadr spec)))]
      [(optional-int64) (if (cadr spec) (number->string (cadr spec)) "None")]
      [(optional-int-array)
       (if (cadr spec) (format "[~a]" (csv (cadr spec))) "None")]
      [(dtype) (if (cadr spec) (format "torch.~a" (cadr spec)) "None")]
      [else (error 'generated-parity "unknown recipe spec: ~a" spec)]))

  (define (spec->python-call-arg spec i)
    (case (car spec)
      [(kwarg) (format "~a=a~a" (cadr spec) i)]
      [(dtype) (format "dtype=a~a" i)]
      [else (format "a~a" i)]))

  (define (generated-python-result py-name specs inplace?)
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

  (define (check-generated-parity entry [specs-override #f] [label ""])
    (define name (car entry))
    (define py-name (cadr entry))
    (define kinds (caddr entry))
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
     (define manifest (with-input-from-file generated-manifest read))
     (for-each check-generated-parity manifest)
     ;; override drives: optional-argument paths the default recipes leave
     ;; absent (or vice versa); the labels name the driven path
     (check-generated-parity
      (assq 'slice-tensor manifest)
      '((tensor 6) (int64 0) (optional-int64 #f) (optional-int64 #f)
        (int64 2))
      "[open]")
     (check-generated-parity
      (assq 'avg-pool2d manifest)
      '((tensor 1 1 4 4) (int-array (2 2)) (int-array (2 2))
        (int-array (0 0)) (bool #f) (bool #t) (optional-int64 2))
      "[divisor=2]")
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
     (check-generated-parity
      (assq 'conv2d manifest)
      '((tensor 1 1 5 5) (tensor 2 1 3 3) (optional-tensor #f)
        (int-array (1 1)) (int-array (0 0)) (int-array (1 1)) (int64 1))
      "[no-bias]")
     (check-generated-parity
      (assq 'sum-dim-intlist manifest)
      '((tensor 2 3) (optional-int-array #f) (bool #f) (dtype #f))
      "[full]")
     (check-generated-parity
      (assq 'mean-dim manifest)
      '((tensor 2 3) (optional-int-array #f) (bool #f) (dtype #f))
      "[full]")
     (check-generated-parity
      (assq 'sum-dim-intlist manifest)
      '((tensor 2 3) (optional-int-array (1)) (bool #t) (dtype #f))
      "[keepdim]")
     (check-generated-parity
      (assq 'mean-dim manifest)
      '((tensor 2 3) (optional-int-array (1)) (bool #t) (dtype #f))
      "[keepdim]")
     (check-generated-parity
      (assq 'layer-norm manifest)
      '((tensor 2 3) (int-array (3)) (optional-tensor #f)
        (optional-tensor #f) (double 1e-5) (bool #t))
      "[no-affine]")
     (check-generated-parity
      (assq 'embedding manifest)
      '((tensor 5 3) (int-tensor (0 2 4 1)) (int64 2) (bool #f) (bool #f))
      "[padding-idx=2]")
     (check-generated-parity
      (assq 'layer-norm manifest)
      '((tensor 2 2 3) (int-array (2 3)) (optional-tensor 2 3)
        (optional-tensor 2 3) (double 1e-5) (bool #t))
      "[multi-dim]")
     (check-generated-parity
      (assq 'tril manifest)
      '((tensor 4 4) (int64 -1))
      "[diag=-1]")
     (check-generated-parity
      (assq 'triu manifest)
      '((tensor 4 4) (int64 1))
      "[diag=1]")]))
