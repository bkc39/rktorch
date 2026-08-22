#lang racket/base

;; raco review lints without macro expansion and cannot see the re-exports.
#|review: ignore|#

(require racket/contract
         "foreign.rkt"
         (only-in "foreign/contracts.rkt" pos-size/c nonneg-size/c)
         "nn/conv.rkt"
         "nn/dropout.rkt"
         "nn/embedding.rkt"
         "nn/init.rkt"
         "nn/layer-norm.rkt"
         "nn/linear.rkt"
         "nn/loss.rkt"
         "nn/module.rkt"
         "nn/optim.rkt"
         "nn/sequential.rkt"
         "nn/state-dict.rkt")

(provide define-module
         gen:module
         module-forward
         module-parameters
         module-named-parameters
         module-buffers
         module-training?
         in-eval-mode)

(provide
 (contract-out
  [module? (-> any/c boolean?)]
  [parameters (-> module? (listof tensor?))]
  [named-parameters (->* (module?) (string?)
                         (listof (cons/c string? tensor?)))]
  [buffers (-> module? (listof tensor?))]
  [forward (-> module? any/c ... any)]
  [train! (-> module? module?)]
  [eval! (-> module? module?)]
  [call-with-eval-mode (-> module? (-> any) any)]
  ;; PascalCase constructors / lowercase predicates and functional ops keep
  ;; `(require torch torch/nn)` collision-free (#11).
  [Linear (-> exact-positive-integer? exact-positive-integer? linear?)]
  [linear? (-> any/c boolean?)]
  [Conv2d (->* (exact-positive-integer? exact-positive-integer? pos-size/c)
               (#:stride pos-size/c #:padding nonneg-size/c)
               conv2d?)]
  [conv2d? (-> any/c boolean?)]
  [MaxPool2d (->* (pos-size/c)
                   (#:stride (or/c #f pos-size/c) #:padding nonneg-size/c)
                   max-pool2d?)]
  [max-pool2d? (-> any/c boolean?)]
  [Flatten (->* () (#:start-dim exact-integer? #:end-dim exact-integer?)
                flatten?)]
  [flatten? (-> any/c boolean?)]
  [Dropout (->* () (#:p (and/c (>=/c 0) (</c 1))) dropout?)]
  [dropout? (-> any/c boolean?)]
  [Embedding (-> exact-positive-integer? exact-positive-integer?
                 embedding?)]
  [embedding? (-> any/c boolean?)]
  [LayerNorm (->* ((or/c exact-positive-integer?
                         (non-empty-listof exact-positive-integer?)))
                  (#:eps real?)
                  layer-norm?)]
  [layer-norm? (-> any/c boolean?)]
  [Sequential (-> module? ... sequential?)]
  [sequential? (-> any/c boolean?)]
  [uniform-init (-> (listof exact-nonnegative-integer?) real? real? tensor?)]
  [normal-init (-> (listof exact-nonnegative-integer?) tensor?)]
  [kaiming-uniform (->* ((listof exact-nonnegative-integer?)) (#:a real?)
                        tensor?)]
  [fan-in (-> (listof exact-nonnegative-integer?)
              exact-nonnegative-integer?)]
  [sgd (-> (listof tensor?) #:lr real? sgd?)]
  [sgd? (-> any/c boolean?)]
  [adam (->* ((listof tensor?))
             (#:lr real? #:beta1 real? #:beta2 real? #:eps real?)
             adam?)]
  [adam? (-> any/c boolean?)]
  [step! (-> optimizer? void?)]
  [zero-grads! (-> optimizer? void?)]
  [mse-loss (-> tensor? tensor? tensor?)]
  [cross-entropy (-> tensor? tensor? tensor?)]
  [state-dict (-> module? (listof (cons/c string? tensor?)))]
  [save-state! (-> module? path-string? void?)]
  [load-state! (-> module? path-string? void?)]))
