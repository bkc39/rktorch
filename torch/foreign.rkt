#lang racket/base

;; raco review lints unexpanded and reads a re-export facade's requires as
;; unused
#|review: ignore|#

(require (only-in racket/contract/base -> contract-out)
         (submod "foreign/device-type.rkt" checked)
         (submod "foreign/error.rkt" checked)
         (only-in "foreign/structs.rkt" tensor-free!)
         (submod "foreign/structs.rkt" checked)
         (except-in "foreign/ops.rkt"
                    device->type+index dims-rest/c dtype/c
                    item to-dtype tensor-dtype to-device tensor-device
                    tensor-shape tensor->list)
         (submod "foreign/ops.rkt" checked)
         (except-in "foreign/creation-ops.rkt" tensor)
         (submod "foreign/creation-ops.rkt" checked)
         (except-in "foreign/tensor-ops.rkt"
                    reshape unsqueeze sum matmul add sub mul div neg)
         (submod "foreign/tensor-ops.rkt" checked)
         "foreign/operators.rkt"
         "foreign/nn-promoted.rkt"
         (except-in "foreign/promoted.rkt" tensor-ref tensor-ref!)
         (submod "foreign/promoted.rkt" checked)
         (only-in "foreign/ref-syntax.rkt" ref ref!)
         (only-in "foreign/sized.rkt" gen:sized length sized?)
         (except-in "foreign/autograd-ops.rkt" requires-grad!)
         (submod "foreign/autograd-ops.rkt" checked)
         (submod "foreign/slice.rkt" checked))

(provide ref ref! with-no-grad with-default-device)

(provide (rename-out [t+ +] [t- -] [t* *] [t/ /])
         @)

;; every name below is contracted at its definition site
(provide torch-version
         manual-seed!
         randn
         rand
         uniform!
         tensor?
         exn:fail:rktorch:oom?
         tensor-shape
         tensor-numel
         tensor->vector
         tensor->list
         tensor->repr
         tensor->string)

(provide zeros
         ones
         full
         zeros-like
         ones-like
         full-like
         randn-like
         rand-like
         make-generator
         generator?
         randperm
         draw-seed
         seed/c
         size/c
         arange
         eye
         tensor)

(provide reshape
         view
         transpose
         (rename-out [transpose t])
         T
         permute
         squeeze
         unsqueeze
         cat
         stack
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
         ::
         slice?)

(provide add
         sub
         mul
         div
         pow
         abs
         neg
         relu
         sigmoid
         gelu
         exp
         log
         sqrt
         tanh
         sin
         cos
         max
         min)

(provide sum
         (rename-out [sum Σ])
         mean
         argmax
         softmax
         log-softmax)

(provide matmul
         mm
         mv
         dot)

(provide conv1d
         conv2d
         max-pool2d
         avg-pool2d
         adaptive-avg-pool2d)

(provide tril
         triu
         masked-fill
         embedding
         layer-norm)

(provide eq
         ne
         lt
         le
         gt
         ge)

(provide item
         to-dtype
         tensor-dtype
         shape
         dtype
         numel
         length
         gen:sized
         sized?)

(provide native-memory-use
         cuda-memory-stats
         cuda-empty-cache!
         mps-empty-cache!
         reclaim-native-memory!
         finalizer-failures
         finalizer-diagnostics)

(provide device
         device?
         device-type
         device-index
         cpu-device
         cuda-device
         cuda-available?
         cuda-if-available
         cuda-device-count
         mps-device
         mps-available?
         mps-if-available
         accelerator-if-available
         set-default-device!
         default-device
         call-with-default-device
         to-device
         tensor-device
         to
         to-able?
         prop:to
         device/c
         dtype/c)

(provide requires-grad!
         requires-grad?
         backward!
         grad
         has-grad?
         maybe-grad
         detach
         grad-enabled?
         call-with-no-grad
         sub!
         zero!
         mul!
         zero-grad!)

(module+ unsafe
  (require (submod "foreign/ops.rkt" unsafe))
  (provide
   to!
   (contract-out
    [tensor-free! (-> tensor? void?)])))
