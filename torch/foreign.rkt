#lang racket/base

;; raco review lints unexpanded, so the contract-out re-exports below would
;; be reported as "provided but not defined".
#|review: ignore|#

(require ffi/vector
         racket/contract
         "foreign/contracts.rkt"
         (except-in "foreign/device-type.rkt" device)
         (only-in "foreign/error.rkt" exn:fail:rktorch:oom?)
         "foreign/structs.rkt"
         "foreign/ops.rkt"
         "foreign/tensor-ops.rkt"
         "foreign/operators.rkt"
         "foreign/nn-promoted.rkt"
         (except-in "foreign/promoted.rkt" tensor-ref tensor-ref!)
         (submod "foreign/promoted.rkt" checked)
         (only-in "foreign/ref-syntax.rkt" ref ref!)
         (except-in "foreign/autograd-ops.rkt" requires-grad!)
         (submod "foreign/autograd-ops.rkt" checked)
         (submod "foreign/slice.rkt" checked))

(provide ref ref! with-no-grad with-default-device)

(provide ::
         slice?
         conv1d
         conv2d
         max-pool2d
         avg-pool2d
         adaptive-avg-pool2d
         tril
         triu
         masked-fill
         embedding
         layer-norm
         flatten
         narrow
         select
         index-select
         masked-select
         nonzero
         take
         gather
         take-along-dim
         where
         tensor-ref
         tensor-ref!
         index-copy!
         index-add!
         index-fill!
         scatter!
         scatter-add!
         masked-fill!
         masked-scatter!
         abs
         sin
         cos
         eq
         ne
         lt
         le
         gt
         ge
         requires-grad!
         requires-grad?
         backward!
         grad
         has-grad?
         maybe-grad
         detach
         grad-enabled?
         call-with-no-grad)

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
  [exn:fail:rktorch:oom? (-> any/c boolean?)]
  [tensor-shape (-> tensor? (listof exact-nonnegative-integer?))]
  [tensor-numel (-> tensor? exact-nonnegative-integer?)]
  [tensor->vector (-> tensor? (or/c f32vector? f64vector? s64vector?))]
  [tensor->list (-> tensor? (listof real?))]
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
  [tensor (->* ((or/c real? list? vector? f32vector? s64vector?))
               (#:requires-grad? boolean?
                #:device (or/c #f device/c)
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
  ;; elementwise
  [add binary-arith/c]
  [sub binary-arith/c]
  [mul binary-arith/c]
  [div binary-arith/c]
  [pow (-> tensor? tensor-or-real/c tensor?)]
  [neg (-> tensor? tensor?)]
  [relu (-> tensor? tensor?)]
  [sigmoid (-> tensor? tensor?)]
  [gelu (-> tensor? tensor?)]
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
  ;; out-marshalling
  [item (-> tensor? real?)]
  [to-dtype (-> tensor? (or/c 'float32 'float64 'int64 'bool) tensor?)]
  [tensor-dtype (-> tensor? (or/c 'float32 'float64 'int64 'bool))]
  [shape (-> tensor? (listof exact-nonnegative-integer?))]
  [dtype (-> tensor? (or/c 'float32 'float64 'int64 'bool))]
  [numel (-> tensor? exact-nonnegative-integer?)]
  ;; memory
  ;; native-memory-use is a handle-attributed estimate: views charge
  ;; their full extents (shared storage double-counts) and
  ;; ATen-internal allocations are absent
  [native-memory-use
   (-> (listof (cons/c device? exact-nonnegative-integer?)))]
  [cuda-memory-stats
   (->* () (device/c)
        (listof (cons/c (or/c 'allocated 'reserved 'peak-allocated)
                        exact-nonnegative-integer?)))]
  [cuda-empty-cache! (-> void?)]
  [mps-empty-cache! (-> void?)]
  [reclaim-native-memory! (-> void?)]
  [finalizer-failures (-> exact-nonnegative-integer?)]
  [finalizer-diagnostics
   (-> (list/c (cons/c 'runs exact-nonnegative-integer?)
               (cons/c 'failures exact-nonnegative-integer?)
               (cons/c 'messages (listof string?))
               (cons/c 'ledger-entries exact-nonnegative-integer?)))]
  ;; device
  [device (->i ([target (or/c tensor? 'cpu 'cuda 'mps)])
               ([index (target)
                       (case target
                         [(cuda) exact-nonnegative-integer?]
                         [(cpu mps) 0]
                         [else none/c])])
               [result device?])]
  [device? (-> any/c boolean?)]
  [device-type (-> device? (or/c 'cpu 'cuda 'mps))]
  [device-index (-> device? exact-nonnegative-integer?)]
  [cpu-device (-> device?)]
  [cuda-device (->* () (exact-nonnegative-integer?) device?)]
  [cuda-available? (-> boolean?)]
  [cuda-if-available (-> device?)]
  [cuda-device-count (-> exact-nonnegative-integer?)]
  [mps-device (-> device?)]
  [mps-available? (-> boolean?)]
  [mps-if-available (-> device?)]
  [accelerator-if-available (-> device?)]
  [set-default-device! (-> device/c void?)]
  [default-device (-> device?)]
  [call-with-default-device (-> device/c (-> any) any)]
  [to-device (-> tensor? device/c tensor?)]
  [tensor-device (-> tensor? device?)]
  ;; in-place
  [sub! (->* (tensor? tensor?) (real?) void?)]
  [zero! (-> tensor? void?)]
  [mul! (-> tensor? real? void?)]
  [zero-grad! (-> tensor? void?)]))

(module+ unsafe
  (provide
   (contract-out
    [tensor-free! (-> tensor? void?)])))
