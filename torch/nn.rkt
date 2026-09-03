#lang racket/base

;; raco review lints without macro expansion and cannot see the re-exports.
#|review: ignore|#

(require racket/contract
         "foreign.rkt"
         "nn/conv.rkt"
         "nn/dropout.rkt"
         "nn/embedding.rkt"
         "nn/init.rkt"
         "nn/layer-norm.rkt"
         "nn/linear.rkt"
         "nn/loss.rkt"
         (except-in "nn/module.rkt" module? named-parameters)
         (submod "nn/module.rkt" checked)
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

;; contracted at their definitions
(provide module?
         parameters
         named-parameters
         buffers
         forward
         train!
         eval!
         call-with-eval-mode
         Dropout
         dropout?
         Sequential
         sequential?)

(provide ctc-loss)

;; PascalCase constructors / lowercase predicates and functional ops keep
;; `(require torch torch/nn)` collision-free (#11).  Each layer is
;; contracted at its definition.
(provide Linear
         linear?
         Conv1d
         conv1d?
         Conv2d
         conv2d?
         MaxPool2d
         max-pool2d?
         Flatten
         flatten?
         Embedding
         embedding?
         LayerNorm
         layer-norm?)

(provide
 (contract-out
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
