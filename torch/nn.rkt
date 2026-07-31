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
         ;; nn layer constructors are PascalCase (Conv2d/MaxPool2d/Flatten/…),
         ;; so they don't collide with the lowercase functional ops on `torch`;
         ;; `(require torch torch/nn)` is clean (#11).
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

;; pos-size/c (positive kernel/stride) and nonneg-size/c (padding may be 0)
;; are shared from foreign/contracts.rkt — see the require above.

;; Macro + generic interface (for hand-written gen:module layers).
;; in-eval-mode is a macro (dynamic-extent form), exported as-is.
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
  ;; train/eval mode (returns the model, like torch.nn.Module.train()/eval())
  [train! (-> module? module?)]
  [eval! (-> module? module?)]
  [call-with-eval-mode (-> module? (-> any) any)]
  ;; layers: PascalCase constructors, lowercase predicates (Racket idiom)
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
  ;; composition
  [Sequential (-> module? ... sequential?)]
  [sequential? (-> any/c boolean?)]
  ;; initializers
  [uniform-init (-> (listof exact-nonnegative-integer?) real? real? tensor?)]
  [normal-init (-> (listof exact-nonnegative-integer?) tensor?)]
  [kaiming-uniform (->* ((listof exact-nonnegative-integer?)) (#:a real?)
                        tensor?)]
  [fan-in (-> (listof exact-nonnegative-integer?)
              exact-nonnegative-integer?)]
  ;; optimizers
  [sgd (-> (listof tensor?) #:lr real? sgd?)]
  [sgd? (-> any/c boolean?)]
  [adam (->* ((listof tensor?))
             (#:lr real? #:beta1 real? #:beta2 real? #:eps real?)
             adam?)]
  [adam? (-> any/c boolean?)]
  [step! (-> optimizer? void?)]
  [zero-grads! (-> optimizer? void?)]
  ;; losses
  [mse-loss (-> tensor? tensor? tensor?)]
  [cross-entropy (-> tensor? tensor? tensor?)]
  ;; safetensors state-dict
  [state-dict (-> module? (listof (cons/c string? tensor?)))]
  [save-state! (-> module? path-string? void?)]
  [load-state! (-> module? path-string? void?)]))
