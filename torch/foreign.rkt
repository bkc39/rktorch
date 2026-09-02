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
         (except-in "foreign/tensor-ops.rkt"
                    reshape unsqueeze tensor sum matmul add sub mul div neg)
         (submod "foreign/tensor-ops.rkt" checked)
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

(provide zeros
         ones
         full
         arange
         eye
         tensor
         reshape
         view
         transpose
         (rename-out [transpose t])
         permute
         squeeze
         unsqueeze
         cat
         stack
         sum
         (rename-out [sum Σ])
         mean
         argmax
         softmax
         log-softmax
         matmul
         mm
         mv
         dot
         add
         sub
         mul
         div
         pow
         neg
         relu
         sigmoid
         gelu
         exp
         log
         sqrt
         tanh
         max
         min)

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
