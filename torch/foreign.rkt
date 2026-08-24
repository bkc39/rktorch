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
         "foreign/promoted.rkt"
         (only-in "foreign/ref-syntax.rkt" ref)
         "foreign/autograd-ops.rkt")

(provide ref with-no-grad with-default-device)

(define bool-tensor/c
  (and/c tensor? (lambda (x) (eq? (tensor-dtype x) 'bool))))

(define int64-tensor/c
  (and/c tensor? (lambda (x) (eq? (tensor-dtype x) 'int64))))

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
  [flatten flatten/c]
  [narrow (-> tensor? index/c index/c exact-positive-integer? tensor?)]
  [select (-> tensor? index/c index/c tensor?)]
  [index-select
   (-> tensor? index/c
       (and/c int64-tensor/c
              (lambda (x) (= 1 (length (tensor-shape x)))))
       tensor?)]
  [masked-select (-> tensor? bool-tensor/c tensor?)]
  [nonzero (-> tensor? tensor?)]
  [take (->i ([v (or/c tensor? list?)]
              [n (v) (if (tensor? v)
                         (or/c int64-tensor/c
                               (listof exact-integer?)
                               (vectorof exact-integer?))
                         exact-nonnegative-integer?)])
             [result (v) (if (tensor? v) tensor? list?)])]
  [gather (-> tensor? index/c int64-tensor/c tensor?)]
  [take-along-dim (->* (tensor? int64-tensor/c) ((or/c #f index/c)) tensor?)]
  [where (case-> (-> bool-tensor/c (listof tensor?))
                 (-> bool-tensor/c tensor? (or/c tensor? real?)
                     tensor?))]
  [tensor-ref (-> tensor? index-spec/c ...
                  (or/c tensor? number? boolean?))]
  [:: (let ([bound/c (or/c #f exact-integer?)])
        (case-> (-> slice?)
                (-> bound/c slice?)
                (-> bound/c bound/c slice?)
                (-> bound/c bound/c exact-integer? slice?)))]
  [slice? (-> any/c boolean?)]
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
  ;; conv + pooling
  [conv2d (->* (tensor? tensor?)
               (#:bias (or/c tensor? #f) #:stride pool-size/c
                #:padding pool-size/c #:dilation pool-size/c
                #:groups index/c)
               tensor?)]
  [max-pool2d (->* (tensor? pool-size/c)
                   (#:stride (or/c pool-size/c #f) #:padding pool-size/c
                    #:dilation pool-size/c #:ceil-mode boolean?)
                   tensor?)]
  [avg-pool2d (->* (tensor? pool-size/c)
                   (#:stride (or/c pool-size/c #f) #:padding pool-size/c
                    #:ceil-mode boolean? #:count-include-pad boolean?
                    #:divisor-override (or/c exact-positive-integer? #f))
                   tensor?)]
  [adaptive-avg-pool2d (-> tensor? pool-size/c tensor?)]
  ;; transformer primitives
  [tril (->* (tensor?) (exact-integer?) tensor?)]
  [triu (->* (tensor?) (exact-integer?) tensor?)]
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
  ;; comparisons
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
  ;; in-place
  [sub! (->* (tensor? tensor?) (real?) void?)]
  [zero! (-> tensor? void?)]
  [mul! (-> tensor? real? void?)]
  [zero-grad! (-> tensor? void?)]))

(module+ unsafe
  (provide
   (contract-out
    [tensor-free! (-> tensor? void?)])))
