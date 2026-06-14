#lang racket/base

;; raco review does surface-level linting without macro expansion, so it cannot
;; see that every `contract-out` entry below re-exports an imported identifier —
;; it would report each as "provided but not defined".  This is a pure
;; re-export facade with nothing else to lint.
#|review: ignore|#

;; Facade for the safe, contracted FFI layer.  `(require torch/foreign)`
;; exposes the low-level surface; all implementation lives in foreign/*.  A
;; `tensor` is a wrapper struct over a cpointer whose underlying handle is
;; GC-reclaimed.
;;
;; `(require (submod torch/foreign unsafe))` additionally exposes
;; `tensor-free!` for callers that need deterministic release.  It is
;; idempotent: a second call hits the cpointer tag guard and raises
;; `exn:fail:contract` instead of double-freeing.
;;
;; Contracts are applied here, at the facade boundary, so this file is the
;; single authoritative description of the public surface.

(require ffi/vector
         racket/contract
         "foreign/contracts.rkt"
         "foreign/structs.rkt"
         "foreign/ops.rkt"
         "foreign/tensor-ops.rkt"
         "foreign/operators.rkt"
         "foreign/promoted.rkt"
         "foreign/autograd-ops.rkt")

;; with-no-grad is a macro (a dynamic-extent form), so it bypasses
;; contract-out; its expansion bottoms out in the contracted procedures.
(provide with-no-grad)

;; Arithmetic operators (+ - * / and matmul's @) shadow racket/base in the
;; rkt-polars style: plain renames rather than contract-out, so the numeric
;; fast path pays no contract overhead; the tensor paths produce the same
;; errors as the named ops they delegate to (add/sub/mul/div/matmul).
(provide (rename-out [t+ +] [t- -] [t* *] [t/ /])
         @)

(provide
 (contract-out
  [torch-version (-> string?)]
  [manual-seed! (-> exact-nonnegative-integer? void?)]
  [randn (->* () #:rest dims-rest/c tensor?)]
  [rand (->* () #:rest dims-rest/c tensor?)]
  [uniform! (-> tensor? real? real? void?)]
  [tensor? (-> any/c boolean?)]
  [tensor-shape (-> tensor? (listof exact-nonnegative-integer?))]
  [tensor-numel (-> tensor? exact-nonnegative-integer?)]
  [tensor->vector (-> tensor? f32vector?)]
  [tensor->list (-> tensor? (listof real?))]
  ;; tensor->repr: the PyTorch `repr` text (what the REPL prints);
  ;; tensor->string: ATen's C++ `operator<<` text.
  [tensor->repr (-> tensor? string?)]
  [tensor->string (-> tensor? string?)]
  ;; creation
  [zeros (->* () #:rest dims-rest/c tensor?)]
  [ones (->* () #:rest dims-rest/c tensor?)]
  [full (->* (real?) #:rest dims-rest/c tensor?)]
  [arange arange/c]
  [eye (->* (exact-nonnegative-integer?)
            (exact-nonnegative-integer?)
            tensor?)]
  [tensor (->* ((or/c real? list?)) (#:requires-grad? boolean?) tensor?)]
  ;; shape
  [reshape (->* (tensor?) #:rest (listof index/c) tensor?)]
  [view (->* (tensor?) #:rest (listof index/c) tensor?)]
  [transpose (-> tensor? index/c index/c tensor?)]
  ;; terse alias, PyTorch-flavored: (t a 0 1) == (transpose a 0 1)
  [rename transpose t (-> tensor? index/c index/c tensor?)]
  [permute (->* (tensor?) #:rest (listof index/c) tensor?)]
  [squeeze (->* (tensor?) (index/c) tensor?)]
  [unsqueeze (-> tensor? index/c tensor?)]
  [cat (->* ((non-empty-listof tensor?)) (index/c) tensor?)]
  [stack (->* ((non-empty-listof tensor?)) (index/c) tensor?)]
  ;; flatten shadows racket/list's: a tensor collapses dims, else defers.
  [flatten flatten/c]
  ;; narrow returns a *view* into `self`: in-place writes to the result
  ;; mutate the original (shared storage; ATen refcount keeps it alive).
  ;; length is positive — ATen rejects a 0-length narrow.
  [narrow (-> tensor? index/c index/c exact-positive-integer? tensor?)]
  ;; elementwise (binary ops take a real on either side, tensor required
  ;; on at least one)
  [add binary-arith/c]
  [sub binary-arith/c]
  [mul binary-arith/c]
  [div binary-arith/c]
  [pow (-> tensor? tensor-or-real/c tensor?)]
  [neg (-> tensor? tensor?)]
  [relu (-> tensor? tensor?)]
  [sigmoid (-> tensor? tensor?)]
  ;; exp/log/sqrt/tanh/max/min shadow racket/base: tensors hit libtorch,
  ;; anything else defers to the racket/base function, so requiring
  ;; torch never breaks numeric code.
  [exp unary-numeric/c]
  [log log/c]
  [sqrt unary-numeric/c]
  [tanh unary-numeric/c]
  [max reduce-or-variadic/c]
  [min reduce-or-variadic/c]
  ;; reductions (Σ is sum's unicode alias: (~> x (* x) Σ))
  [sum (-> tensor? tensor?)]
  [rename sum Σ (-> tensor? tensor?)]
  [mean (-> tensor? tensor?)]
  ;; argmax shadows racket/list's: (argmax proc lst) delegates to it.
  [argmax argmax/c]
  [softmax (-> tensor? index/c tensor?)]
  [log-softmax (-> tensor? index/c tensor?)]
  ;; linalg
  [matmul (-> tensor? tensor? tensor?)]
  [mm (-> tensor? tensor? tensor?)]
  [mv (-> tensor? tensor? tensor?)]
  [dot (-> tensor? tensor? tensor?)]
  ;; conv + pooling (promoted from the generated surface, PyTorch-style
  ;; keyword defaults; stride/padding/dilation take an int or an [h w] list)
  [conv2d (->* (tensor? tensor?)
               (#:bias (or/c tensor? #f) #:stride pool-size/c
                #:padding pool-size/c #:dilation pool-size/c
                #:groups index/c)
               tensor?)]
  ;; #:stride #f means "default to kernel-size" (PyTorch's stride=None).
  [max-pool2d (->* (tensor? pool-size/c)
                   (#:stride (or/c pool-size/c #f) #:padding pool-size/c
                    #:dilation pool-size/c #:ceil-mode boolean?)
                   tensor?)]
  ;; #:stride #f means "default to kernel-size" (PyTorch's stride=None).
  [avg-pool2d (->* (tensor? pool-size/c)
                   (#:stride (or/c pool-size/c #f) #:padding pool-size/c
                    #:ceil-mode boolean? #:count-include-pad boolean?
                    ;; positive: a 0 divisor is a divide-by-zero in ATen.
                    #:divisor-override (or/c exact-positive-integer? #f))
                   tensor?)]
  [adaptive-avg-pool2d (-> tensor? pool-size/c tensor?)]
  ;; comparisons (tensor lhs, tensor-or-real rhs) -> float32 masks
  [eq compare/c]
  [ne compare/c]
  [lt compare/c]
  [le compare/c]
  [gt compare/c]
  [ge compare/c]
  ;; out-marshalling
  [item (-> tensor? real?)]
  [to-dtype (-> tensor? (or/c 'float32 'float64 'int64) tensor?)]
  ;; autograd
  [requires-grad! (->* (tensor?) (boolean?) tensor?)]
  [requires-grad? (-> tensor? boolean?)]
  [backward! (-> tensor? void?)]
  [grad (-> tensor? tensor?)]
  [has-grad? (-> tensor? boolean?)]
  [maybe-grad (-> tensor? (or/c tensor? #f))]
  [detach (-> tensor? tensor?)]
  [grad-enabled? (-> boolean?)]
  [call-with-no-grad (-> (-> any) any)]
  ;; in-place update primitives (use under with-no-grad, like torch.optim)
  [sub! (->* (tensor? tensor?) (real?) void?)]
  [zero! (-> tensor? void?)]
  [mul! (-> tensor? real? void?)]
  [zero-grad! (-> tensor? void?)]))

;; Explicit-free helper for deterministic release.  See the file comment.
(module+ unsafe
  (provide
   (contract-out
    [tensor-free! (-> tensor? void?)])))
