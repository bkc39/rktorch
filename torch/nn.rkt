#lang racket/base

;; raco review lints unexpanded and reads a re-export facade's requires as
;; unused
#|review: ignore|#

(require "nn/conv.rkt"
         "nn/dropout.rkt"
         "nn/embedding.rkt"
         (submod "nn/init.rkt" checked)
         "nn/layer-norm.rkt"
         "nn/linear.rkt"
         "nn/loss.rkt"
         (except-in "nn/module.rkt" module? named-parameters)
         (submod "nn/module.rkt" checked)
         (only-in "nn/optim.rkt" adam adam? sgd sgd? step! zero-grads!)
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

;; every name below is contracted at its definition site
(provide module?
         parameters
         named-parameters
         buffers
         forward
         train!
         eval!
         call-with-eval-mode)

;; PascalCase constructors / lowercase predicates and functional ops keep
;; `(require torch torch/nn)` collision-free (#11).
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
         Dropout
         dropout?
         Embedding
         embedding?
         LayerNorm
         layer-norm?
         Sequential
         sequential?)

(provide uniform-init
         normal-init
         kaiming-uniform
         fan-in)

(provide sgd
         sgd?
         adam
         adam?
         step!
         zero-grads!)

(provide mse-loss
         cross-entropy
         ctc-loss)

(provide state-dict
         save-state!
         load-state!)
