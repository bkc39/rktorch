#lang racket/base

;; raco review lints unexpanded and reads a re-export facade's requires as
;; unused
#|review: ignore|#

(require "nn/batch-norm.rkt"
         (except-in "nn/buffer.rkt" Buffer?)
         (submod "nn/buffer.rkt" checked)
         "nn/clip.rkt"
         "nn/conv.rkt"
         "nn/dropout.rkt"
         "nn/ema.rkt"
         "nn/embedding.rkt"
         "nn/group-norm.rkt"
         (submod "nn/init.rkt" checked)
         "nn/layer-hash.rkt"
         "nn/layer-list.rkt"
         "nn/layer-norm.rkt"
         (except-in "nn/layer.rkt"
                    child-name/c
                    children-by-index
                    children-by-key
                    parameters-by-key
                    in-layers
                    layer?
                    named-buffers
                    named-parameters
                    step/c
                    training?)
         (submod "nn/layer.rkt" checked)
         "nn/linear.rkt"
         "nn/loss.rkt"
         (only-in "nn/optim.rkt"
                  adam adam? learning-rate rmsprop rmsprop?
                  set-learning-rate! sgd sgd? step! zero-grads!)
         (submod "nn/optim.rkt" checked)
         "nn/scheduler.rkt"
         (except-in "nn/parameter.rkt" Parameter Parameter?)
         (submod "nn/parameter.rkt" checked)
         "nn/recurrent.rkt"
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
         layer-mode
         layer-set-mode!
         with-mode
         in-mode
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
         set-mode!
         layer-training?
         call-with-mode
         call-with-eval-mode
         mode/c
         training?
         evaluating?)

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
         parameters-by-key
         Parameters?
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
         ConvTranspose2d
         conv-transpose2d?
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
         GroupNorm
         group-norm?
         BatchNorm2d
         batch-norm2d?
         BatchNorm1d
         batch-norm1d?
         Sequential
         sequential?
         LSTM
         lstm?
         GRU
         gru?)

(provide uniform-init
         normal-init
         kaiming-uniform
         fan-in)

(provide sgd
         sgd?
         adam
         adam?
         rmsprop
         rmsprop?
         optimizer?
         step!
         zero-grads!
         learning-rate
         set-learning-rate!)

(provide scheduler?
         scheduler-step-count
         scheduler-rate
         scheduler-optimizer-of
         step-lr
         multi-step-lr
         exponential-lr
         cosine-annealing-lr
         linear-lr
         one-cycle-lr
         lambda-lr)

(provide clip-grad-norm!)

(provide ema
         ema?
         ema-average
         ema-decay
         ema-update!)

(provide mse-loss
         cross-entropy
         ctc-loss
         binary-cross-entropy-with-logits
         huber-loss
         l1-loss
         nll-loss)

(provide state-dict
         save-state!
         load-state!)
