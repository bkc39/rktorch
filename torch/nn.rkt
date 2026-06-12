#lang racket/base

;; raco review does surface-level linting without macro expansion, so it cannot
;; see that every `contract-out` entry below re-exports an imported identifier.
#|review: ignore|#

;; Facade for the nn layer: `(require torch/nn)` beside `(require
;; torch)`, mirroring `import torch.nn`. The module system is
;; `define-module` (Python-style field registration, done at expansion time)
;; over the `gen:module` interface; hand-written layers implement gen:module
;; directly. Models are plain struct trees owned by the GC — no global
;; parameter store.

(require racket/contract
         "foreign.rkt"
         "nn/init.rkt"
         "nn/linear.rkt"
         "nn/loss.rkt"
         "nn/module.rkt"
         "nn/optim.rkt")

;; Macro + generic interface (for hand-written gen:module layers).
(provide define-module
         gen:module
         module-forward
         module-parameters
         module-named-parameters
         module-buffers)

(provide
 (contract-out
  [module? (-> any/c boolean?)]
  [parameters (-> module? (listof tensor?))]
  [named-parameters (->* (module?) (string?)
                         (listof (cons/c string? tensor?)))]
  [buffers (-> module? (listof tensor?))]
  [forward (->* (module?) #:rest (listof any/c) any)]
  ;; layers
  [linear (-> exact-positive-integer? exact-positive-integer? linear?)]
  [linear? (-> any/c boolean?)]
  ;; initializers
  [uniform-init (-> (listof exact-nonnegative-integer?) real? real? tensor?)]
  [kaiming-uniform (->* ((listof exact-nonnegative-integer?)) (#:a real?)
                        tensor?)]
  [fan-in (-> (listof exact-nonnegative-integer?)
              exact-nonnegative-integer?)]
  ;; optimizer
  [sgd (-> (listof tensor?) #:lr real? sgd?)]
  [sgd? (-> any/c boolean?)]
  [step! (-> sgd? void?)]
  [zero-grads! (-> sgd? void?)]
  ;; losses
  [mse-loss (-> tensor? tensor? tensor?)]))
