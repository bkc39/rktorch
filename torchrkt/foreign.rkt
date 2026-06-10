#lang racket/base

;; raco review does surface-level linting without macro expansion, so it cannot
;; see that every `contract-out` entry below re-exports an imported identifier —
;; it would report each as "provided but not defined".  This is a pure
;; re-export facade with nothing else to lint.
#|review: ignore|#

;; Facade for the safe, contracted FFI layer.  `(require torchrkt/foreign)`
;; exposes the low-level surface; all implementation lives in foreign/*.  A
;; `tensor` is a wrapper struct over a cpointer whose underlying handle is
;; GC-reclaimed.
;;
;; `(require (submod torchrkt/foreign unsafe))` additionally exposes
;; `tensor-free!` for callers that need deterministic release.  It is
;; idempotent: a second call hits the cpointer tag guard and raises
;; `exn:fail:contract` instead of double-freeing.
;;
;; Contracts are applied here, at the facade boundary, so this file is the
;; single authoritative description of the public surface.

(require ffi/vector
         racket/contract
         "foreign/structs.rkt"
         "foreign/ops.rkt"
         "foreign/tensor-ops.rkt"
         "foreign/autograd-ops.rkt")

;; with-no-grad is a macro (a dynamic-extent form), so it bypasses
;; contract-out; its expansion bottoms out in the contracted procedures.
(provide with-no-grad)

;; Shape arguments and the variadic creation ops.
(define dims-rest/c (listof exact-nonnegative-integer?))
;; reshape/view accept -1 ("infer this dimension"), so plain integers.
(define index/c exact-integer?)
(define tensor-or-real/c (or/c tensor? real?))

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
  [arange (case-> (-> real? tensor?)
                  (-> real? real? tensor?)
                  (-> real? real? real? tensor?))]
  [eye (->* (exact-nonnegative-integer?)
            (exact-nonnegative-integer?)
            tensor?)]
  [tensor (-> (or/c real? list?) tensor?)]
  ;; shape
  [reshape (->* (tensor?) #:rest (listof index/c) tensor?)]
  [view (->* (tensor?) #:rest (listof index/c) tensor?)]
  [transpose (-> tensor? index/c index/c tensor?)]
  [permute (->* (tensor?) #:rest (listof index/c) tensor?)]
  [squeeze (->* (tensor?) (index/c) tensor?)]
  [unsqueeze (-> tensor? index/c tensor?)]
  [cat (->* ((non-empty-listof tensor?)) (index/c) tensor?)]
  [stack (->* ((non-empty-listof tensor?)) (index/c) tensor?)]
  ;; elementwise (binary ops take a real on either side)
  [add (-> tensor-or-real/c tensor-or-real/c tensor?)]
  [sub (-> tensor-or-real/c tensor-or-real/c tensor?)]
  [mul (-> tensor-or-real/c tensor-or-real/c tensor?)]
  [div (-> tensor-or-real/c tensor-or-real/c tensor?)]
  [pow (-> tensor? tensor-or-real/c tensor?)]
  [neg (-> tensor? tensor?)]
  [relu (-> tensor? tensor?)]
  [sigmoid (-> tensor? tensor?)]
  [tanh (-> tensor? tensor?)]
  ;; exp/log/sqrt/max/min shadow racket/base: tensors hit libtorch, anything
  ;; else defers to the racket/base function, so requiring torchrkt is safe.
  [exp (-> (or/c tensor? number?) (or/c tensor? number?))]
  [log (->* ((or/c tensor? number?)) (number?) (or/c tensor? number?))]
  [sqrt (-> (or/c tensor? number?) (or/c tensor? number?))]
  [max (->* (tensor-or-real/c) #:rest (listof real?) tensor-or-real/c)]
  [min (->* (tensor-or-real/c) #:rest (listof real?) tensor-or-real/c)]
  ;; reductions
  [sum (-> tensor? tensor?)]
  [mean (-> tensor? tensor?)]
  ;; argmax shadows racket/list's: (argmax proc lst) delegates to it.
  [argmax (->* ((or/c tensor? procedure?))
               ((or/c index/c list?) #:keepdim boolean?)
               any/c)]
  [softmax (-> tensor? index/c tensor?)]
  [log-softmax (-> tensor? index/c tensor?)]
  ;; linalg
  [matmul (-> tensor? tensor? tensor?)]
  [mm (-> tensor? tensor? tensor?)]
  [mv (-> tensor? tensor? tensor?)]
  [dot (-> tensor? tensor? tensor?)]
  ;; out-marshalling
  [item (-> tensor? real?)]
  [to-dtype (-> tensor? (or/c 'float32 'float64 'int64) tensor?)]
  ;; autograd
  [requires-grad! (->* (tensor?) (boolean?) tensor?)]
  [requires-grad? (-> tensor? boolean?)]
  [backward! (-> tensor? void?)]
  [grad (-> tensor? tensor?)]
  [has-grad? (-> tensor? boolean?)]
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
