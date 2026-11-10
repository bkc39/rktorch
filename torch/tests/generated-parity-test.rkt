#lang racket/base

;; Run: raco test torch/tests/generated-parity-test.rkt (inside `nix develop`,
;; which provides Python torch; SKIPS when python3 can't import torch).

(module+ test
  (require (only-in json jsexpr->string)
           (only-in racket/match match-define)
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in racket/string string-join)
           rackunit
           "../main.rkt"
           "private/python-env.rkt")

  (define-runtime-path generated-manifest "generated-parity.rktd")
  (define-runtime-path generated-rkt "../generated.rkt")

  ;; Every op in the codegen manifest needs an input recipe here (a missing
  ;; one fails loudly). Spec -> input: (tensor dim ...) = seeded randn,
  ;; (tensors ...) a list of them, (index-tensors (i ...) #f ...) int64 index
  ;; tensors or None, (bool-tensor ...) a genuine bool handle
  ;; via `ne 0`, (kwarg "name" v) a scalar (or none) passed as name=v to a kwarg-only
  ;; aten arg, (optional-scalar v) a Scalar? bound, (uniform-tensor dim ...)
  ;; a seeded rand (non-negative, for probabilities), (uniform-tensors ...)
  ;; a list of them, (step-tensors v ...) 0-dim float32 tensors holding each
  ;; v, (scalar-list v ...) a list of reals, other heads are
  ;; literals with #f = None. A recipe of 'device-only marks an op with no CPU
  ;; kernel, which a device-guarded test drives instead.
  (define fused-adam-recipe
    '((tensors (3) (2 2)) (tensors (3) (2 2)) (tensors (3) (2 2))
      (uniform-tensors (3) (2 2)) (tensors) (step-tensors 1.0 3.0)
      (kwarg "lr" 0.01) (kwarg "beta1" 0.9) (kwarg "beta2" 0.999)
      (kwarg "weight_decay" 0.01) (kwarg "eps" 1e-8)
      (kwarg "amsgrad" #f) (kwarg "maximize" #f)
      (kwarg "grad_scale" none) (kwarg "found_inf" none)))

  (define generated-recipes
    (hash 'matmul '((tensor 2 3) (tensor 3 2))
          'mm '((tensor 2 2) (tensor 2 2))
          'mv '((tensor 2 3) (tensor 3))
          'dot '((tensor 4) (tensor 4))
          'abs-tensor '((tensor 2 3))
          'broadcast-to '((tensor 1 3) (int-array (2 3)))
          'cos-tensor '((tensor 2 3))
          'sin-tensor '((tensor 2 3))
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
          'index-tensor '((tensor 3 4) (index-tensors #f (3 1 3)))
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
          'conv1d '((tensor 1 2 8) (tensor 3 2 3) (optional-tensor 3)
                    (int-array (1)) (int-array (0)) (int-array (1))
                    (int64 1))
          'ctc-loss-intlist '((tensor 4 2 5) (int-tensor-2d ((1 2) (3 4)))
                              (int-array (4 4)) (int-array (2 2)) (int64 0)
                              (int64 1) (bool #f))
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
          'triu '((tensor 4 4) (int64 0))
          'conv-transpose2d-input '((tensor 1 2 4 4) (tensor 2 3 3 3)
                                    (optional-tensor 3) (int-array (1 1))
                                    (int-array (0 0)) (int-array (0 0))
                                    (int64 1) (int-array (1 1)))
          'group-norm '((tensor 2 4 3 3) (int64 2) (optional-tensor 4)
                        (optional-tensor 4) (double 1e-5) (bool #t))
          'silu '((tensor 2 3))
          'clamp '((tensor 2 3) (optional-scalar -0.5) (optional-scalar 0.5))
          'repeat-interleave-self-int '((tensor 2 3) (int64 2) (optional-int64 1)
                                        (kwarg "output_size" none))
          ;; running_var must stay positive, so it is drawn as ones
          'batch-norm '((tensor 2 3 4 4) (optional-tensor 3) (optional-tensor 3)
                        (optional-tensor 3) (optional-tensor-ones 3) (bool #f)
                        (double 0.1) (double 1e-5) (bool #t))
          'leaky-relu '((tensor 2 3) (double 0.01))
          'binary-cross-entropy-with-logits '((tensor 2 3) (tensor 2 3)
                                              (optional-tensor #f)
                                              (optional-tensor #f) (int64 1))
          'huber-loss '((tensor 2 3) (tensor 2 3) (int64 1) (double 1.0))
          'l1-loss '((tensor 2 3) (tensor 2 3) (int64 1))
          'flip '((tensor 2 3) (int-array (1)))
          'linear '((tensor 4 3) (tensor 2 3) (optional-tensor 2))
          'topk '((tensor 3 5) (int64 2) (int64 -1) (bool #t) (bool #t))
          'sort-tensor '((tensor 3 5) (int64 -1) (bool #f))
          'cudnn-rnn-flatten-weight 'device-only
          'argsort '((tensor 3 5) (int64 -1) (bool #f))
          'multinomial '((uniform-tensor 3 6) (int64 4) (bool #f)
                         (kwarg "generator" none))
          'lstm-input '((tensor 5 2 3) (tensors (1 2 4) (1 2 4))
                        (tensors (16 3) (16 4) (16) (16))
                        (bool #t) (int64 1) (double 0.0) (bool #f) (bool #f)
                        (bool #f))
          'gru-input '((tensor 5 2 3) (tensor 1 2 4)
                       (tensors (12 3) (12 4) (12) (12))
                       (bool #t) (int64 1) (double 0.0) (bool #f) (bool #f)
                       (bool #f))
          'scaled-dot-product-attention '((tensor 2 3 4) (tensor 2 5 4)
                                          (tensor 2 5 6) (optional-tensor #f)
                                          (double 0.0) (bool #f)
                                          (kwarg "scale" none)
                                          (kwarg "enable_gqa" #f))
          'add-tensor '((tensor 3) (tensor 3) (kwarg "alpha" 0.5))
          'div-scalar! '((tensor 3) (double 0.7))
          'lerp-scalar! '((tensor 3) (tensor 3) (double 0.1))
          'foreach-add-scalar! '((tensors (3) (2 2)) (double 0.25))
          'foreach-add-list! '((tensors (3) (2 2)) (tensors (3) (2 2))
                               (kwarg "alpha" -0.5))
          'foreach-mul-scalar! '((tensors (3) (2 2)) (double 0.9))
          'foreach-div-scalarlist! '((tensors (3) (2 2)) (scalar-list 0.5 3.0))
          'foreach-lerp-scalar! '((tensors (3) (2 2)) (tensors (3) (2 2))
                                  (double 0.1))
          'foreach-addcmul-scalar! '((tensors (3) (2 2)) (tensors (3) (2 2))
                                     (tensors (3) (2 2)) (double 0.001))
          'foreach-addcdiv-scalar! '((tensors (3) (2 2)) (tensors (3) (2 2))
                                     (uniform-tensors (3) (2 2)) (double -0.01))
          'foreach-addcdiv-scalarlist! '((tensors (3) (2 2)) (tensors (3) (2 2))
                                         (uniform-tensors (3) (2 2))
                                         (scalar-list -0.01 -0.02))
          'foreach-sqrt! '((uniform-tensors (3) (2 2)))
          'foreach-copy! '((tensors (3) (2 2)) (tensors (3) (2 2)) (bool #f))
          'fused-adam! fused-adam-recipe
          'fused-adamw! fused-adam-recipe
          'fused-sgd! '((tensors (3) (2 2)) (tensors (3) (2 2)) (tensors (3) (2 2))
                        (kwarg "weight_decay" 5e-4) (kwarg "momentum" 0.9)
                        (kwarg "lr" 0.1) (kwarg "dampening" 0.0)
                        (kwarg "nesterov" #t) (kwarg "maximize" #f)
                        (kwarg "is_first_step" #f) (kwarg "grad_scale" none)
                        (kwarg "found_inf" none))
          'affine-grid-generator '((tensor 2 2 3) (int-array (2 3 4 5)) (bool #f))
          'grid-sampler-2d '((tensor 2 3 4 5) (tensor 2 6 7 2) (int64 0)
                             (int64 0) (bool #f))
          'upsample-bicubic2d '((tensor 1 2 3 3) (int-array (5 4)) (bool #f)
                                (kwarg "scales_h" none) (kwarg "scales_w" none))
          'floor-tensor '((tensor 2 3))
          'fmod-scalar '((tensor 2 3) (double 1.0))))

  ;; Tensor specs draw seeded randns left to right — both sides consume the
  ;; same RNG stream, so spec order and draw counts must match exactly.
  (define (spec->racket-arg spec)
    (case (car spec)
      [(tensor) (apply randn (cdr spec))]
      [(uniform-tensor) (apply rand (cdr spec))]
      [(tensors)
       (for/list ([dims (in-list (cdr spec))])
         (apply randn dims))]
      [(uniform-tensors)
       (for/list ([dims (in-list (cdr spec))])
         (apply rand dims))]
      [(step-tensors) (map tensor (cdr spec))]
      [(scalar-list) (cdr spec)]
      [(optional-tensor)
       (if (equal? (cdr spec) '(#f)) #f (apply randn (cdr spec)))]
      [(optional-tensor-ones) (apply ones (cdr spec))]
      [(int-tensor int-tensor-2d) (~> (cadr spec) tensor (to-dtype 'int64))]
      [(index-tensors)
       (for/list ([positions (in-list (cdr spec))])
         (and positions (to-dtype (tensor positions) 'int64)))]
      [(scalar-tensor) (tensor (cadr spec))]
      [(bool-tensor) (~> (cadr spec) tensor (ne 0))]
      [(int64 double bool int-array optional-int64 optional-int-array
        optional-scalar dtype)
       (cadr spec)]
      [(kwarg) (let ([v (caddr spec)]) (if (eq? v 'none) #f v))]
      [else (error 'generated-parity "unknown recipe spec: ~a" spec)]))

  (define (spec->python-expr spec)
    (define (csv vs)
      (string-join (map number->string vs) ", "))
    (case (car spec)
      [(tensor) (format "torch.randn(~a)" (csv (cdr spec)))]
      [(uniform-tensor) (format "torch.rand(~a)" (csv (cdr spec)))]
      [(tensors)
       (format "[~a]"
               (string-join (for/list ([dims (in-list (cdr spec))])
                              (format "torch.randn(~a)" (csv dims)))
                            ", "))]
      [(uniform-tensors)
       (format "[~a]"
               (string-join (for/list ([dims (in-list (cdr spec))])
                              (format "torch.rand(~a)" (csv dims)))
                            ", "))]
      [(step-tensors)
       (format "[~a]"
               (string-join (for/list ([v (in-list (cdr spec))])
                              (format "torch.tensor(~a)" v))
                            ", "))]
      [(scalar-list) (format "[~a]" (csv (cdr spec)))]
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
      [(index-tensors)
       (format "[~a]"
               (string-join
                (for/list ([positions (in-list (cdr spec))])
                  (if positions
                      (format "torch.tensor([~a], dtype=torch.int64)"
                              (csv positions))
                      "None"))
                ", "))]
      [(scalar-tensor) (format "torch.tensor(~a)" (cadr spec))]
      [(bool-tensor)
       (format "torch.tensor([~a], dtype=torch.bool)" (csv (cadr spec)))]
      [(int64 double) (number->string (cadr spec))]
      [(kwarg)
       (define v (caddr spec))
       (cond
         [(eq? v 'none) "None"]
         [(boolean? v) (if v "True" "False")]
         [else (number->string v)])]
      [(bool) (if (cadr spec) "True" "False")]
      [(int-array) (format "[~a]" (csv (cadr spec)))]
      [(optional-int64 optional-scalar)
       (if (cadr spec) (number->string (cadr spec)) "None")]
      [(optional-int-array)
       (if (cadr spec) (format "[~a]" (csv (cadr spec))) "None")]
      [(dtype) (if (cadr spec) (format "torch.~a" (cadr spec)) "None")]
      [else (error 'generated-parity "unknown recipe spec: ~a" spec)]))

  (define (spec->python-call-arg spec i)
    (case (car spec)
      [(kwarg) (format "~a=a~a" (cadr spec) i)]
      [(dtype) (format "dtype=a~a" i)]
      [else (format "a~a" i)]))

  (define (mutable-positions kinds)
    (for/list ([k (in-list kinds)] [i (in-naturals)]
               #:when (eq? k 'mutable-tensor-list))
      i))

  (define (parity-check entry [specs-override #f] [label ""])
    (define name (car entry))
    (list entry
          (or specs-override
              (hash-ref generated-recipes name
                        (lambda ()
                          (error 'generated-parity
                                 "no input recipe for generated op ~a; add one to ~a"
                                 name "generated-parity-test.rkt"))))
          label))

  (define (python-snippet check)
    (match-define (list (list _ py-name kinds inplace? _) specs _) check)
    (define mutables (mutable-positions kinds))
    (define call-args
      (string-join (for/list ([s (in-list specs)] [i (in-naturals)])
                     (spec->python-call-arg s i))
                   ", "))
    (define invoke (format "torch.ops.aten.~a(~a)" py-name call-args))
    (string-append
     (apply string-append
            (for/list ([s (in-list specs)]
                       [i (in-naturals)])
              (format "a~a = ~a\n" i (spec->python-expr s))))
     (cond
       [(pair? mutables)
        (format "~a\nrs = [t for a in (~a) for t in a]\n" invoke
                (string-join (for/list ([i (in-list mutables)])
                               (format "a~a," i))
                             " "))]
       [inplace? (format "~a\nrs = (a0,)\n" invoke)]
       [else (format "r = ~a\nrs = r if isinstance(r, tuple) else (r,)\n"
                     invoke)])))

  (define (python-results checks)
    (parameterize ([current-input-port
                    (~> checks (map python-snippet _) jsexpr->string open-input-string)])
      (python-check "generated_parity.py")))

  (define (check-generated-parity check expected)
    (match-define (list (list name _ kinds _ returns) specs label) check)
    (cond
      [(hash? expected)
       (fail (format "~a~a: python failed: ~a" name label (hash-ref expected 'error)))]
      [else
       (check-equal? (length specs) (length kinds)
                     (format "~a~a: recipe arity matches manifest" name label))
       (define mutables (mutable-positions kinds))
       (manual-seed! 0)
       (define args (map spec->racket-arg specs))
       (define op (dynamic-require generated-rkt name))
       (define answered (call-with-values (lambda () (apply op args)) list))
       (define results
         (if (zero? returns)
             (apply append (map (lambda (i) (list-ref args i)) mutables))
             answered))
       (check-equal? (if (zero? returns) (length mutables) (length answered))
                     (if (zero? returns) (length mutables) returns)
                     (format "~a~a: return count matches manifest" name label))
       (check-equal? (length results) (length expected)
                     (format "~a~a: return count parity" name label))
       (for ([result (in-list results)]
             [j (in-list expected)]
             [k (in-naturals)])
         (check-equal? (tensor-shape result) (hash-ref j 'shape)
                       (format "~a~a: generated shape parity, return ~a"
                               name label k))
         (for ([r (in-list (tensor->list result))]
               [p (in-list (hash-ref j 'values))]
               [i (in-naturals)])
           (check-= r p tol
                    (format "~a~a: generated value ~a parity, return ~a"
                            name label i k))))]))

  (cond
    [(not (python-torch-available?))
     (printf "[generated-parity-test] skipped: python3 `torch` not available ~a\n"
             "(run inside `nix develop`)")]
    [else
     (define manifest (with-input-from-file generated-manifest read))
     (define checks
       (append
        (for/list ([entry (in-list manifest)]
                   #:unless (eq? 'device-only
                                 (hash-ref generated-recipes (car entry) #f)))
          (parity-check entry))
        (list
         ;; override drives: optional-argument paths the default recipes leave
         ;; absent (or vice versa); the labels name the driven path
         (parity-check
          (assq 'slice-tensor manifest)
          '((tensor 6) (int64 0) (optional-int64 #f) (optional-int64 #f)
            (int64 2))
          "[open]")
         (parity-check
          (assq 'avg-pool2d manifest)
          '((tensor 1 1 4 4) (int-array (2 2)) (int-array (2 2))
            (int-array (0 0)) (bool #f) (bool #t) (optional-int64 2))
          "[divisor=2]")
         (parity-check
          (assq 'nll-loss manifest)
          '((tensor 4 3) (int-tensor (0 2 1 0)) (optional-tensor-ones 3)
            (int64 1) (int64 -100))
          "[weight]")
         (parity-check
          (assq 'cross-entropy-loss manifest)
          '((tensor 4 3) (int-tensor (0 2 1 0)) (optional-tensor-ones 3)
            (int64 1) (int64 -100) (double 0.0))
          "[weight]")
         (parity-check
          (assq 'nll-loss manifest)
          '((tensor 4 3) (int-tensor (0 2 1 0)) (optional-tensor #f)
            (int64 0) (int64 -100))
          "[none]")
         (parity-check
          (assq 'cross-entropy-loss manifest)
          '((tensor 4 3) (int-tensor (0 2 1 0)) (optional-tensor #f)
            (int64 0) (int64 -100) (double 0.0))
          "[none]")
         (parity-check
          (assq 'conv2d manifest)
          '((tensor 1 1 5 5) (tensor 2 1 3 3) (optional-tensor #f)
            (int-array (1 1)) (int-array (0 0)) (int-array (1 1)) (int64 1))
          "[no-bias]")
         (parity-check
          (assq 'sum-dim-intlist manifest)
          '((tensor 2 3) (optional-int-array #f) (bool #f) (dtype #f))
          "[full]")
         (parity-check
          (assq 'mean-dim manifest)
          '((tensor 2 3) (optional-int-array #f) (bool #f) (dtype #f))
          "[full]")
         (parity-check
          (assq 'sum-dim-intlist manifest)
          '((tensor 2 3) (optional-int-array (1)) (bool #t) (dtype #f))
          "[keepdim]")
         (parity-check
          (assq 'mean-dim manifest)
          '((tensor 2 3) (optional-int-array (1)) (bool #t) (dtype #f))
          "[keepdim]")
         (parity-check
          (assq 'layer-norm manifest)
          '((tensor 2 3) (int-array (3)) (optional-tensor #f)
            (optional-tensor #f) (double 1e-5) (bool #t))
          "[no-affine]")
         (parity-check
          (assq 'embedding manifest)
          '((tensor 5 3) (int-tensor (0 2 4 1)) (int64 2) (bool #f) (bool #f))
          "[padding-idx=2]")
         (parity-check
          (assq 'layer-norm manifest)
          '((tensor 2 2 3) (int-array (2 3)) (optional-tensor 2 3)
            (optional-tensor 2 3) (double 1e-5) (bool #t))
          "[multi-dim]")
         (parity-check
          (assq 'tril manifest)
          '((tensor 4 4) (int64 -1))
          "[diag=-1]")
         (parity-check
          (assq 'triu manifest)
          '((tensor 4 4) (int64 1))
          "[diag=1]")
         (parity-check
          (assq 'conv-transpose2d-input manifest)
          '((tensor 1 2 4 4) (tensor 2 3 3 3) (optional-tensor #f)
            (int-array (1 1)) (int-array (0 0)) (int-array (0 0)) (int64 1)
            (int-array (1 1)))
          "[no-bias]")
         (parity-check
          (assq 'conv-transpose2d-input manifest)
          '((tensor 1 2 4 4) (tensor 2 3 3 3) (optional-tensor 3)
            (int-array (2 2)) (int-array (0 0)) (int-array (1 1)) (int64 1)
            (int-array (1 1)))
          "[stride-2+output-padding]")
         (parity-check
          (assq 'conv-transpose2d-input manifest)
          '((tensor 1 4 4 4) (tensor 4 3 3 3) (optional-tensor 6)
            (int-array (1 1)) (int-array (0 0)) (int-array (0 0)) (int64 2)
            (int-array (2 2)))
          "[groups-2+dilation-2]")
         (parity-check
          (assq 'group-norm manifest)
          '((tensor 2 4 3 3) (int64 2) (optional-tensor #f) (optional-tensor #f)
            (double 1e-5) (bool #t))
          "[no-affine]")
         (parity-check
          (assq 'group-norm manifest)
          '((tensor 2 4 3 3) (int64 1) (optional-tensor 4) (optional-tensor 4)
            (double 1e-5) (bool #t))
          "[one-group]")
         (parity-check
          (assq 'clamp manifest)
          '((tensor 2 3) (optional-scalar -0.5) (optional-scalar #f))
          "[min-only]")
         (parity-check
          (assq 'clamp manifest)
          '((tensor 2 3) (optional-scalar #f) (optional-scalar 0.5))
          "[max-only]")
         (parity-check
          (assq 'repeat-interleave-self-int manifest)
          '((tensor 2 3) (int64 3) (optional-int64 #f) (kwarg "output_size" none))
          "[flattened]")
         (parity-check
          (assq 'repeat-interleave-self-int manifest)
          '((tensor 2 3) (int64 3) (optional-int64 #f) (kwarg "output_size" 18))
          "[flattened+output-size]")
         (parity-check
          (assq 'batch-norm manifest)
          '((tensor 2 3 4 4) (optional-tensor 3) (optional-tensor 3)
            (optional-tensor 3) (optional-tensor-ones 3) (bool #t) (double 0.1)
            (double 1e-5) (bool #t))
          "[training]")
         (parity-check
          (assq 'batch-norm manifest)
          '((tensor 2 3 4 4) (optional-tensor #f) (optional-tensor #f)
            (optional-tensor #f) (optional-tensor #f) (bool #t) (double 0.1)
            (double 1e-5) (bool #t))
          "[training+no-stats+no-affine]")
         (parity-check
          (assq 'leaky-relu manifest)
          '((tensor 2 3) (double 0.2))
          "[slope-0.2]")
         (parity-check
          (assq 'binary-cross-entropy-with-logits manifest)
          '((tensor 2 3) (tensor 2 3) (optional-tensor 3) (optional-tensor-ones 3)
            (int64 2))
          "[weighted+sum]")
         (parity-check
          (assq 'huber-loss manifest)
          '((tensor 2 3) (tensor 2 3) (int64 0) (double 0.5))
          "[none+delta-0.5]")
         (parity-check
          (assq 'l1-loss manifest)
          '((tensor 2 3) (tensor 2 3) (int64 2))
          "[sum]")
         (parity-check
          (assq 'flip manifest)
          '((tensor 2 3) (int-array (0 1)))
          "[both-dims]")
         (parity-check
          (assq 'linear manifest)
          '((tensor 4 3) (tensor 2 3) (optional-tensor #f))
          "[no-bias]")
         (parity-check
          (assq 'topk manifest)
          '((tensor 3 5) (int64 3) (int64 0) (bool #f) (bool #t))
          "[smallest-along-dim-0]")
         (parity-check
          (assq 'sort-tensor manifest)
          '((tensor 3 5) (int64 0) (bool #t))
          "[descending-dim-0]")
         (parity-check
          (assq 'argsort manifest)
          '((tensor 3 5) (int64 0) (bool #t))
          "[descending-dim-0]")
         (parity-check
          (assq 'multinomial manifest)
          '((uniform-tensor 3 6) (int64 8) (bool #t) (kwarg "generator" none))
          "[replacement]")
         (parity-check
          (assq 'lstm-input manifest)
          '((tensor 2 5 3) (tensors (4 2 4) (4 2 4))
            (tensors (16 3) (16 4) (16 3) (16 4)
                     (16 8) (16 4) (16 8) (16 4))
            (bool #f) (int64 2) (double 0.0) (bool #f) (bool #t) (bool #t))
          "[2-layer-bidirectional-batch-first-no-bias]")
         (parity-check
          (assq 'gru-input manifest)
          '((tensor 2 5 3) (tensor 4 2 4)
            (tensors (12 3) (12 4) (12 3) (12 4)
                     (12 8) (12 4) (12 8) (12 4))
            (bool #f) (int64 2) (double 0.0) (bool #f) (bool #t) (bool #t))
          "[2-layer-bidirectional-batch-first-no-bias]")
         (parity-check
          (assq 'scaled-dot-product-attention manifest)
          '((tensor 2 3 4) (tensor 2 5 4) (tensor 2 5 6) (optional-tensor #f)
            (double 0.0) (bool #f) (kwarg "scale" 0.5) (kwarg "enable_gqa" #f))
          "[scale=0.5]")
         (parity-check
          (assq 'scaled-dot-product-attention manifest)
          '((tensor 2 3 4) (tensor 2 5 4) (tensor 2 5 6) (optional-tensor 3 5)
            (double 0.0) (bool #f) (kwarg "scale" none) (kwarg "enable_gqa" #f))
          "[float-mask]")
         (parity-check
          (assq 'scaled-dot-product-attention manifest)
          '((tensor 2 3 4) (tensor 2 5 4) (tensor 2 5 6)
            (bool-tensor (1 0 1 1 0)) (double 0.0) (bool #f) (kwarg "scale" none)
            (kwarg "enable_gqa" #f))
          "[bool-mask]")
         (parity-check
          (assq 'scaled-dot-product-attention manifest)
          '((tensor 2 5 4) (tensor 2 5 4) (tensor 2 5 6) (optional-tensor #f)
            (double 0.0) (bool #t) (kwarg "scale" none) (kwarg "enable_gqa" #f))
          "[causal]")
         (parity-check
          (assq 'scaled-dot-product-attention manifest)
          '((tensor 2 4 3 8) (tensor 2 2 5 8) (tensor 2 2 5 8)
            (optional-tensor #f) (double 0.0) (bool #f) (kwarg "scale" none)
            (kwarg "enable_gqa" #t))
          "[grouped-query]")
         (parity-check
          (assq 'grid-sampler-2d manifest)
          '((tensor 2 3 4 5) (tensor 2 6 7 2) (int64 1) (int64 1) (bool #t))
          "[nearest-border-aligned]")
         (parity-check
          (assq 'grid-sampler-2d manifest)
          '((tensor 2 3 4 5) (tensor 2 6 7 2) (int64 2) (int64 2) (bool #f))
          "[bicubic-reflection]")
         (parity-check
          (assq 'affine-grid-generator manifest)
          '((tensor 2 2 3) (int-array (2 3 4 5)) (bool #t))
          "[aligned]")
         (parity-check
          (assq 'upsample-bicubic2d manifest)
          '((tensor 1 2 8 8) (int-array (4 4)) (bool #t) (kwarg "scales_h" none)
            (kwarg "scales_w" none))
          "[downscale-aligned]"))))
     (define expected (python-results checks))
     (check-equal? (length expected) (length checks) "one python result per case")
     (for-each check-generated-parity checks expected)]))
