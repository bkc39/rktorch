#lang racket/base

;; raco review lints without macro expansion, so every `contract-out`
;; re-export below would be reported as "provided but not defined".
#|review: ignore|#

;; Facade for the safe, contracted FFI layer; implementation lives in
;; foreign/*.  Contracts are applied here, so this file is the single
;; authoritative description of the public surface.  The `unsafe` submodule
;; adds `tensor-free!` for deterministic release.

(require ffi/vector
         racket/contract
         "foreign/contracts.rkt"
         ;; ops.rkt's hybrid `device` supersedes the bare struct constructor
         (except-in "foreign/device-type.rkt" device)
         (only-in "foreign/error.rkt" exn:fail:rktorch:oom?)
         "foreign/structs.rkt"
         "foreign/ops.rkt"
         "foreign/tensor-ops.rkt"
         "foreign/operators.rkt"
         "foreign/promoted.rkt"
         "foreign/autograd-ops.rkt")

;; with-no-grad / with-default-device are macros (dynamic-extent forms), so they
;; bypass contract-out; their expansions bottom out in the contracted procedures.
(provide with-no-grad with-default-device)

;; Plain renames rather than contract-out, so the numeric fast path pays no
;; contract overhead; the tensor paths error like the ops they delegate to.
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
  ;; catch OOM by this type, never by regexing exn messages
  [exn:fail:rktorch:oom? (-> any/c boolean?)]
  [tensor-shape (-> tensor? (listof exact-nonnegative-integer?))]
  [tensor-numel (-> tensor? exact-nonnegative-integer?)]
  [tensor->vector (-> tensor? (or/c f32vector? f64vector? s64vector?))]
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
  ;; data may nest lists/vectors arbitrarily, like torch.tensor; a
  ;; matching-dtype f32vector/s64vector at top level ingests copy-free
  [tensor (->* ((or/c real? list? vector? f32vector? s64vector?))
               (#:requires-grad? boolean?
                #:device (or/c #f device/c)
                ;; inferred int64 for all-exact-integer data, else float32
                #:dtype (or/c #f 'float32 'int64))
               tensor?)]
  ;; shape
  [reshape (-> tensor? index/c ... tensor?)]
  [view (-> tensor? index/c ... tensor?)]
  [transpose (-> tensor? index/c index/c tensor?)]
  [rename transpose t (-> tensor? index/c index/c tensor?)]
  [permute (-> tensor? index/c ... tensor?)]
  [squeeze (->* (tensor?) (index/c) tensor?)]
  [unsqueeze (-> tensor? index/c tensor?)]
  [cat (->* ((non-empty-listof tensor?)) (index/c) tensor?)]
  [stack (->* ((non-empty-listof tensor?)) (index/c) tensor?)]
  ;; flatten shadows racket/list's: a tensor collapses dims, else defers.
  [flatten flatten/c]
  ;; returns a *view* aliasing self; length is positive (ATen rejects 0)
  [narrow (-> tensor? index/c index/c exact-positive-integer? tensor?)]
  ;; elementwise
  [add binary-arith/c]
  [sub binary-arith/c]
  [mul binary-arith/c]
  [div binary-arith/c]
  [pow (-> tensor? tensor-or-real/c tensor?)]
  [neg (-> tensor? tensor?)]
  [relu (-> tensor? tensor?)]
  [sigmoid (-> tensor? tensor?)]
  ;; exact (erf-based) gelu, approximate='none'
  [gelu (-> tensor? tensor?)]
  ;; exp/log/sqrt/tanh/max/min shadow racket/base: tensors hit libtorch,
  ;; anything else defers, so requiring torch never breaks numeric code.
  [exp unary-numeric/c]
  [log log/c]
  [sqrt unary-numeric/c]
  [tanh unary-numeric/c]
  [max reduce-or-variadic/c]
  [min reduce-or-variadic/c]
  ;; reductions
  [sum (-> tensor? tensor?)]
  [rename sum Σ (-> tensor? tensor?)]
  [mean (-> tensor? tensor?)]
  [argmax argmax/c]
  [softmax (-> tensor? index/c tensor?)]
  [log-softmax (-> tensor? index/c tensor?)]
  ;; linalg
  [matmul (-> tensor? tensor? tensor?)]
  [mm (-> tensor? tensor? tensor?)]
  [mv (-> tensor? tensor? tensor?)]
  [dot (-> tensor? tensor? tensor?)]
  ;; conv + pooling (PyTorch-style keyword defaults)
  [conv2d (->* (tensor? tensor?)
               (#:bias (or/c tensor? #f) #:stride pool-size/c
                #:padding pool-size/c #:dilation pool-size/c
                #:groups index/c)
               tensor?)]
  ;; pooling #:stride #f means "default to kernel-size" (PyTorch stride=None)
  [max-pool2d (->* (tensor? pool-size/c)
                   (#:stride (or/c pool-size/c #f) #:padding pool-size/c
                    #:dilation pool-size/c #:ceil-mode boolean?)
                   tensor?)]
  [avg-pool2d (->* (tensor? pool-size/c)
                   (#:stride (or/c pool-size/c #f) #:padding pool-size/c
                    #:ceil-mode boolean? #:count-include-pad boolean?
                    ;; positive: a 0 divisor is a divide-by-zero in ATen
                    #:divisor-override (or/c exact-positive-integer? #f))
                   tensor?)]
  [adaptive-avg-pool2d (-> tensor? pool-size/c tensor?)]
  ;; transformer primitives
  [tril (->* (tensor?) (exact-integer?) tensor?)]
  [triu (->* (tensor?) (exact-integer?) tensor?)]
  ;; mask must be a bool tensor (a comparison result); value may be -inf.0
  [masked-fill (-> tensor? tensor? real? tensor?)]
  [embedding (->* (tensor? tensor?)
                  (#:padding-idx (or/c #f exact-nonnegative-integer?))
                  tensor?)]
  [layer-norm (->* (tensor?
                    (or/c exact-positive-integer?
                          (non-empty-listof exact-positive-integer?)))
                   (#:weight (or/c tensor? #f)
                    #:bias (or/c tensor? #f)
                    #:eps real?)
                   tensor?)]
  ;; comparisons -> bool masks whose *values* read back as float32 (the
  ;; handles stay bool; masked-fill consumes them directly)
  [eq compare/c]
  [ne compare/c]
  [lt compare/c]
  [le compare/c]
  [gt compare/c]
  [ge compare/c]
  ;; out-marshalling
  [item (-> tensor? real?)]
  [to-dtype (-> tensor? (or/c 'float32 'float64 'int64 'bool) tensor?)]
  [tensor-dtype (-> tensor? (or/c 'float32 'float64 'int64 'bool))]
  ;; PyTorch-property short names; the tensor- forms stay as aliases
  [shape (-> tensor? (listof exact-nonnegative-integer?))]
  [dtype (-> tensor? (or/c 'float32 'float64 'int64 'bool))]
  [numel (-> tensor? exact-nonnegative-integer?)]
  ;; live handle-attributed bytes per device, folded from the accounting
  ;; ledger — not total device usage (see raw/memory.rkt)
  [native-memory-use
   (-> (listof (cons/c device? exact-nonnegative-integer?)))]
  ;; the CUDA caching allocator's own gauges — what the ALLOCATOR holds,
  ;; complementing the ledger's what-our-handles-hold view
  [cuda-memory-stats
   (->* () (device/c)
        (listof (cons/c (or/c 'allocated 'reserved 'peak-allocated)
                        exact-nonnegative-integer?)))]
  [cuda-empty-cache! (-> void?)]
  ;; collect -> drain the finalizer executor -> empty the CUDA cache: the
  ;; release-everything-now sequence (see ops.rkt)
  [reclaim-native-memory! (-> void?)]
  ;; guarded-finalizer swallows since startup — silent by design (a
  ;; finalizer has nowhere to raise); growing count = leaking handles
  [finalizer-failures (-> exact-nonnegative-integer?)]
  ;; device placement. hybrid device: query a tensor, or construct from a
  ;; type symbol + optional ordinal; the dependent contract permits the
  ;; ordinal ONLY for 'cuda — (device t 1) and (device 'cpu 1) are
  ;; boundary violations, not internal errors.
  [device (->i ([target (or/c tensor? 'cpu 'cuda)])
               ([index (target)
                       (case target
                         [(cuda) exact-nonnegative-integer?]
                         [(cpu) 0]
                         [else none/c])])
               [result device?])]
  [device? (-> any/c boolean?)]
  [device-type (-> device? (or/c 'cpu 'cuda))]
  [device-index (-> device? exact-nonnegative-integer?)]
  [cpu-device (-> device?)]
  [cuda-device (->* () (exact-nonnegative-integer?) device?)]
  [cuda-available? (-> boolean?)]
  [cuda-if-available (-> device?)]
  [cuda-device-count (-> exact-nonnegative-integer?)]
  [set-default-device! (-> device/c void?)]
  [default-device (-> device?)]
  [call-with-default-device (-> device/c (-> any) any)]
  [to-device (-> tensor? device/c tensor?)]
  [tensor-device (-> tensor? device?)]
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

(module+ unsafe
  (provide
   (contract-out
    [tensor-free! (-> tensor? void?)])))
