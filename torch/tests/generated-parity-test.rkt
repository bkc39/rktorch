#lang racket/base

;; Run: raco test torch/tests/generated-parity-test.rkt (inside `nix develop`,
;; which provides Python torch; SKIPS when python3 can't import torch).

(module+ test
  (require (only-in json jsexpr->string)
           (only-in racket/match match match-define)
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in racket/string string-join)
           rackunit
           "../main.rkt"
           "private/python-env.rkt")

  (define-runtime-path generated-manifest "generated-parity.rktd")
  (define-runtime-path generated-overrides "generated-parity-overrides.rktd")
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
          'fmod-scalar '((tensor 2 3) (double 1.0))
          'maximum '((tensor 2 3) (tensor 2 3))
          'minimum '((tensor 2 3) (tensor 2 3))))

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

  (struct parity-case (entry specs label))

  (define (parity-check entry [specs-override #f] [label ""])
    (match-define (cons name _) entry)
    (parity-case entry
                 (or specs-override
                     (hash-ref generated-recipes name
                               (lambda ()
                                 (error 'generated-parity
                                        "no input recipe for generated op ~a; add one to ~a"
                                        name "generated-parity-test.rkt"))))
                 label))

  (define (python-snippet check)
    (match-define (parity-case (list _ py-name kinds inplace? _) specs _) check)
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

  (define (python-results snippets)
    (parameterize ([current-input-port (~> snippets jsexpr->string open-input-string)])
      (python-check "generated_parity.py")))

  (define (compare-results checks expected #:fail [fail* fail] #:check [check-one check-results])
    (cond
      [(= (length expected) (length checks))
       (for ([check (in-list checks)]
             [result (in-list expected)])
         (match-define (parity-case (cons name _) _ label) check)
         (if (hash? result)
             (fail* (format "~a~a: python failed: ~a" name label (hash-ref result 'error)))
             (check-one check result)))]
      [else
       (fail* (format "one python result per case: ~a for ~a cases"
                      (length expected) (length checks)))]))

  (define (check-results check expected)
    (match-define (parity-case (list name _ kinds _ returns) specs label) check)
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
                         name label i k)))))

  (test-case "a python error fails its case alone, and a short result list fails once"
    (define checks (list (parity-case '(abs "abs" (tensor) #f 1) '((tensor 2)) "[a]")
                         (parity-case '(neg "neg" (tensor) #f 1) '((tensor 2)) "")))
    (define (failures-of expected)
      (define failed '())
      (define checked '())
      (compare-results checks expected
                       #:fail (lambda (why) (set! failed (cons why failed)))
                       #:check (lambda (check _) (set! checked (cons check checked))))
      (list (reverse failed) (length checked)))
    (check-equal? (failures-of (list (hasheq 'error "RuntimeError: boom") '()))
                  '(("abs[a]: python failed: RuntimeError: boom") 1))
    (check-equal? (failures-of (list '()))
                  '(("one python result per case: 1 for 2 cases") 0)))

  (cond
    [(not (python-torch-available?))
     (printf "[generated-parity-test] skipped: python3 `torch` not available ~a\n"
             "(run inside `nix develop`)")]
    [else
     (test-case "the driver reports a snippet whose returns are not tensors and runs the next"
       (match (python-results '("rs = (1,)" "rs = (torch.zeros(1),)"))
         [(list (? hash? failed) second)
          (check-regexp-match #rx"AttributeError" (hash-ref failed 'error))
          (check-equal? second (list (hasheq 'shape '(1) 'values '(0.0))))]
         [results (fail (format "expected an error then a result: ~s" results))]))
     (define manifest (with-input-from-file generated-manifest read))
     (define checks
       (append
        (for/list ([entry (in-list manifest)]
                   #:unless (eq? 'device-only
                                 (hash-ref generated-recipes (car entry) #f)))
          (parity-check entry))
        (for/list ([drive (in-list (with-input-from-file generated-overrides read))])
          (match-define (list name specs label) drive)
          (parity-check (assq name manifest) specs label))))
     (compare-results checks (python-results (map python-snippet checks)))]))
