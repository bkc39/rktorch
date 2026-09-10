#lang racket/base

;; raco review lints unexpanded and reads a re-export facade's requires as
;; unused
#|review: ignore|#

(require (except-in "nn/buffer.rkt" Buffer?)
         (submod "nn/buffer.rkt" checked)
         "nn/conv.rkt"
         "nn/dropout.rkt"
         "nn/embedding.rkt"
         (submod "nn/init.rkt" checked)
         "nn/layer-hash.rkt"
         "nn/layer-list.rkt"
         "nn/layer-norm.rkt"
         (except-in "nn/layer.rkt"
                    child-name/c
                    children-by-index
                    children-by-key
                    in-layers
                    layer?
                    named-buffers
                    named-parameters
                    step/c)
         (submod "nn/layer.rkt" checked)
         "nn/linear.rkt"
         "nn/loss.rkt"
         (only-in "nn/optim.rkt" adam adam? sgd sgd? step! zero-grads!)
         (except-in "nn/parameter.rkt" Parameter Parameter?)
         (submod "nn/parameter.rkt" checked)
         "nn/sequential.rkt"
         "nn/state-dict.rkt")

(provide define-layer
         gen:layer
         layer-forward
         layer-parameters
         layer-named-parameters
         layer-buffers
         layer-named-buffers
         layer-named-children
         layer-training?
         in-eval-mode)

(provide layer?
         procedure->Layer
         parameters
         named-parameters
         buffers
         named-buffers
         children
         named-children
         forward
         train!
         eval!
         call-with-eval-mode)

(provide Parameter
         Parameter?
         Buffer
         Buffer?
         LayerList
         layer-list?
         LayerHash
         layer-hash?
         children-by-index
         children-by-key
         Children?
         child-ref
         child-name/c
         in-layers
         step/c)

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
